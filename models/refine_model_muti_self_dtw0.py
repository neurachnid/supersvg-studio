import torch.nn as nn
import torch.nn.functional as F
import torch
import pydiffvg
from torchvision.utils import save_image
from util import refine_mophology as mophology
# from util import dpw
from util.refine_soft_dtw_cuda import SoftDTW
import random
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
torch.multiprocessing.set_start_method('spawn', force=True)

class CombinedModel(nn.Module):
    def __init__(self, coarse_model, diff_model,path_num,width):
        super(CombinedModel, self).__init__()
        # self.dtw= SoftDTW(use_cuda=True, gamma=0.1)
        self.coarse_model = coarse_model
        self.diff_model = diff_model
        self.path_num = path_num
        self.width = width
        self.dtw= SoftDTW(use_cuda=True, gamma=0.1)
    def forward(self, x,mask):
        strokes,difference_map,pred_0 = self.refine_model(x,mask)
        mse_loss_0 = F.mse_loss(pred_0*mask,x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        features = strokes.permute(0,2,1).clone().detach()
        strokes = self.diff_model(features,difference_map)
        pred1 = self.rendering(strokes)[:, :3, :, :]
        mse_loss_1 = F.mse_loss(pred1*mask,x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        
        difference_map2 = pred1 - x*mask
        features = strokes.permute(0,2,1).clone().detach()
        strokes2 = self.diff_model(features,difference_map2)
        pred2 = self.rendering(strokes2)[:, :3, :, :]
        mse_loss_2 = F.mse_loss(pred2*mask,x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        
        difference_map3 = pred2 - x*mask
        features = strokes2.permute(0,2,1).clone().detach()
        strokes3 = self.diff_model(features,difference_map3)
        pred3 = self.rendering(strokes3)[:, :3, :, :]
        mse_loss_3 = F.mse_loss(pred3*mask,x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        
        difference_map4 = pred3 - x*mask
        features = strokes3.permute(0,2,1).clone().detach()
        strokes4 = self.diff_model(features,difference_map4)
        pred4 = self.rendering(strokes4)[:, :3, :, :]
        mse_loss_4 = F.mse_loss(pred4*mask,x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        
        return mse_loss_0,mse_loss_1,mse_loss_2,mse_loss_3,mse_loss_4

    def multi_scale_mse(self,x,y):
        assert x.shape == y.shape, "x and y must have the same shape"

        total_loss = 0.0
        scale_factor = 1.0
        down_factor=4
        while x.shape[2] > 1 and x.shape[3] > 1:
            mse_loss = ((x-y)**2).mean(dim=(1,2,3))
            total_loss += scale_factor * mse_loss

            x = F.avg_pool2d(x, kernel_size=down_factor, stride=down_factor, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=down_factor, stride=down_factor, ceil_mode=True)

            scale_factor /= 2

        if x.shape[2] > 0 and x.shape[3] > 0:
            total_loss += scale_factor * ((x-y)**2).mean(dim=(1,2,3))

        return total_loss
    
    def loss(self,x,mask,canvas,epoch_id,iter_id,critic=None,num_loss=False):

        pred_strokes = self.refine_model(x = x*mask-(1-mask),canvas = canvas*mask-(1-mask))

        pred_RGBA = self.rendering(pred_strokes)
        new_pred,alpha = pred_RGBA[:, :3, :, :],pred_RGBA[:, 3:, :, :]

        pred=new_pred*alpha+(1-alpha)*canvas

        loss_pixel = self.multi_scale_mse(pred * mask, x * mask) / (mask.sum(dim=(1,2,3)) / (mask.size(-1) * mask.size(-2)))
        loss_pixel=loss_pixel.mean()

        log_loss = {}

        lambda_mask = max(0.05 - 0.005 * (epoch_id+1), 0.01)
    
        mask = mophology.dilation(mask,m=2)
        loss_mask=((alpha*(1-mask)).sum())/((1-mask).sum())*lambda_mask
        loss=loss_pixel+loss_mask
        log_loss['loss_pixel'] = loss_pixel.item()

        log_loss['loss_mask']=loss_mask.item()

        log_loss["loss"] = loss.item()
        if iter_id%100==0:
            save_images=[x[:8]*mask[:8],canvas[:8]*mask[:8],pred[:8],new_pred[:8],alpha[:8].repeat(1,3,1,1)]
            save_images=torch.cat(save_images,dim=0)
            return loss,log_loss,{'save_images':save_images}
        else:
            return loss,log_loss,{}
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