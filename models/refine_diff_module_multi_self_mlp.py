import torch.nn as nn
import torch
import timm
from pathlib import Path

from util.refine_cross_attention import CrossAttentionBlock
from timm.models.vision_transformer import Block
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .refine_encoder import StrokeAttentionPredictor
from .render import FCN
# from .render_oil import FCNOil
import pydiffvg
import lpips
from torch.multiprocessing import Process, Queue
from util import refine_SVR_render as SVR_render
from torchvision import transforms
import random
from torchvision.utils import save_image
from itertools import repeat
import collections.abc
import time
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
torch.multiprocessing.set_start_method('spawn', force=True)
class FusionConvNet(nn.Module):
    def __init__(self,input_dim=6):
        super(FusionConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.GELU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.GELU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
        self.relu3 = nn.GELU()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))  
        x = self.relu2(self.conv2(x))  
        x = self.relu3(self.conv3(x)) 
        return x


class DiffAddModel(nn.Module):
    def __init__(self, stroke_num=8, stroke_dim=27, hidden_dim=768, self_attn_depth=4, add_canvas=False):
        super(DiffAddModel, self).__init__()
        self.fusion_conv = FusionConvNet(input_dim=9)
        self.add_canvas = add_canvas
        self.diff_feature_extractor1 = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        local_dino = Path(__file__).resolve().parents[1] / "weights" / "dino_deitsmall16_pretrain.pth"
        self.diff_feature_extractor1.load_state_dict(torch.load(str(local_dino), map_location="cpu"))
        self.cross_attn_block = CrossAttentionBlock(x_dim=self.diff_feature_extractor1.embed_dim, y_dim=hidden_dim,
                                                    num_heads=min(stroke_num, 8))
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=hidden_dim, num_heads=min(stroke_num, 8), mlp_ratio=4., qkv_bias=True,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        self.linear_proj = nn.Linear(1, hidden_dim)
        self.linear_head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def extract_features(self, x):
        x = self.diff_feature_extractor1.patch_embed(x)
        cls_token = self.diff_feature_extractor1.cls_token.expand(x.shape[0], -1,
                                                                  -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.diff_feature_extractor1.pos_drop(x + self.diff_feature_extractor1.pos_embed)
        x = self.diff_feature_extractor1.blocks(x)
        x = self.diff_feature_extractor1.norm(x)
        return x[:, 1:]

    def forward(self, x, difference_map, img, canvas=None):
        if canvas is None:
            combined_features = torch.cat((img, difference_map), dim=1)
            combined_features = self.fusion_conv(combined_features)
        else:
            combined_features = torch.cat((img, difference_map, canvas), dim=1)
            combined_features = self.fusion_conv(combined_features)
        batch_size = x.shape[0]
        stroke_dim = x.shape[2]
        stroke_num = x.shape[1]
        diff_features = self.extract_features(combined_features)
        y = x.view(batch_size, stroke_dim * stroke_num, 1)
        y = self.linear_proj(y)  # b,27*32,768
        residual = self.cross_attn_block(diff_features, y)
        residual = self.self_attn_blocks(residual)
        residual = self.linear_head(residual)
        residual = residual.reshape(batch_size, stroke_num, stroke_dim)
        return torch.sigmoid(residual)


class AttnPainterSVG(nn.Module):
    def __init__(self, stroke_num=32, path_num=4, width=128, control_num=False, self_attn_depth=1, num_loss=False,
                 refine=False, return_stroke_num=None,supersvg_version=False):
        super(AttnPainterSVG, self).__init__()
        self.control_num = control_num
        self.path_num = path_num
        self.encoder = StrokeAttentionPredictor(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3,
                                                control_num=control_num, self_attn_depth=self_attn_depth,
                                                num_loss=num_loss,supersvg_version=supersvg_version)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        self.render = SVR_render.SVGObject(size=(width, width))
        self.width = width
        self.loss_fn_vgg = None
        self.fuser_conv = FusionConvNet()
        if return_stroke_num is None:
            self.return_stroke_num = stroke_num
        else:
            self.return_stroke_num = return_stroke_num
            

    def forward(self, x,mask=None,canvas = None,**kwargs):
        if canvas is None:
            strokes = self.encoder(x*mask-(1-mask))    
            pred = self.rendering(strokes, **kwargs)[:, :3, :, :]
            difference_map = pred - x*mask
            return strokes,difference_map,pred
        else:
            in_f = torch.cat((x, canvas), dim=1)     
            f = self.fuser_conv(in_f)
            strokes = self.encoder(f)
            return strokes[:,:self.return_stroke_num]
            
    def predict_path(self, x, num=None,**kwargs):
        if self.control_num:
            if num is None:
                num=random.randint(1,64)
            strokes = self.encoder(x,num)
        else:
            strokes = self.encoder(x)

        return strokes

    def rendering(self, strokes, save_svg_path=None, idx=None):
        imgs = []
        if strokes.size(-1)==27:
            strokes = torch.cat([strokes, torch.ones(strokes.size(0), strokes.size(1), 1).to(strokes.device)], dim=2)
        strokes=strokes.float()
        num_control_points = [2] * self.path_num
        for b in range(strokes.size(0)):
            shapes = []
            groups = []
            for num in range(strokes.size(1)):
                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=strokes[b][num][:-4].reshape(-1, 2) * self.width,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True))
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([num]),
                        fill_color=strokes[b][num][-4:]))
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
            _render = pydiffvg.RenderFunction.apply
            img = _render(self.width, self.width, 2, 2, 0, None, *scene_args)
            imgs.append(img.permute(2, 0, 1))
        imgs = torch.stack(imgs, dim=0)
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
          
        return imgs



