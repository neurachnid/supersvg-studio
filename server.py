import base64
import json
import subprocess
import sys
import tempfile
import os
import signal
import shutil
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
MIN_FREE_DISK_BYTES = int(float(os.environ.get("SUPERSVG_MIN_FREE_GB", "2")) * 1024**3)
CHECKPOINT_DIR = Path(os.environ.get("SUPERSVG_CKPT_DIR", ROOT / "weights"))
REQUIRE_CUDA = os.environ.get("SUPERSVG_REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"}

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError(
        "SUPERSVG_REQUIRE_CUDA is enabled, but PyTorch cannot access an NVIDIA GPU. "
        "Start the container with GPU access (for example, docker run --gpus all)."
    )

app = FastAPI(title="SuperSVG Studio")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ActiveJobController:
    def __init__(self):
        self._lock = threading.Lock()
        self._job_id = 0
        self._process = None

    @staticmethod
    def terminate(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def begin(self):
        with self._lock:
            self._job_id += 1
            job_id = self._job_id
            previous = self._process
            self._process = None
        self.terminate(previous)
        return job_id

    def attach(self, job_id, process):
        with self._lock:
            current = job_id == self._job_id
            if current:
                self._process = process
        if not current:
            self.terminate(process)
        return current

    def is_current(self, job_id):
        with self._lock:
            return job_id == self._job_id

    def finish(self, job_id, process):
        with self._lock:
            if job_id == self._job_id and self._process is process:
                self._process = None


active_job = ActiveJobController()


@app.middleware("http")
async def disable_development_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class VectorizeRequest(BaseModel):
    image: str
    mime_type: str = "image/png"
    path_num: int = Field(default=1000, ge=64, le=4000)
    optimize_iter: int = Field(default=0, ge=0, le=100)
    refine_paths_per_segment: int = Field(default=8, ge=1, le=32)
    refine_batch_size: int = Field(default=64, ge=1, le=128)
    coarse_paths_per_segment: int = Field(default=64, ge=16, le=256)
    coarse_margin: int = Field(default=2, ge=0, le=32)
    refine_margin: int = Field(default=0, ge=0, le=32)
    working_resolution: int = Field(default=512, ge=256, le=1024)
    coarse_compactness: float = Field(default=50.0, ge=0.1, le=200.0)
    refine_compactness: float = Field(default=20.0, ge=0.1, le=200.0)
    slic_sigma: float = Field(default=5.0, ge=0.0, le=20.0)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.01)
    path_penalty: float = Field(default=0.000001, ge=0.0, le=0.001)
    seed: int = Field(default=0, ge=0, le=999999)
    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "cuda_required": REQUIRE_CUDA,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "free_disk_bytes": shutil.disk_usage("/tmp").free,
    }


@app.post("/api/vectorize")
def vectorize(payload: VectorizeRequest):
    if REQUIRE_CUDA and payload.device != "cuda":
        raise HTTPException(status_code=422, detail="This deployment requires CUDA; CPU inference is disabled.")
    free_disk = min(shutil.disk_usage("/tmp").free, shutil.disk_usage(CHECKPOINT_DIR.parent).free)
    if free_disk < MIN_FREE_DISK_BYTES:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough free disk space to start vectorization. "
                f"At least {MIN_FREE_DISK_BYTES / 1024**3:.1f} GB is required."
            ),
        )
    try:
        encoded = payload.image.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Images are limited to 25 MB.")
    if not torch.cuda.is_available() and payload.path_num * (payload.optimize_iter + 1) > 80000:
        safe_passes = max(0, 80000 // payload.path_num - 1)
        raise HTTPException(
            status_code=422,
            detail=(
                f"This CPU-only host cannot safely fine-tune {payload.path_num:,} paths "
                f"for {payload.optimize_iter} passes. Use {safe_passes} passes or fewer, "
                "reduce the path count, or run with CUDA."
            ),
        )

    suffix = ALLOWED_MIME_TYPES.get(payload.mime_type, ".png")
    job_id = active_job.begin()

    def stream():
        with tempfile.TemporaryDirectory(prefix="supersvg_", dir="/tmp") as temp_dir:
            if not active_job.is_current(job_id):
                yield json.dumps({"type": "cancelled", "message": "Superseded by a newer request."}) + "\n"
                return

            temp_path = Path(temp_dir)
            input_path = temp_path / f"input{suffix}"
            output_dir = temp_path / "output"
            input_path.write_bytes(image_bytes)

            device = payload.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            command = [
                sys.executable,
                "-u",
                str(ROOT / "inference.py"),
                "--input_path",
                str(input_path),
                "--output_dir",
                str(output_dir),
                "--device",
                device,
                "--ckpt_dir",
                str(CHECKPOINT_DIR),
                "--path_num",
                str(payload.path_num),
                "--optimize_iter",
                str(payload.optimize_iter),
                "--refine_paths_per_segment",
                str(payload.refine_paths_per_segment),
                "--refine_batch_size",
                str(payload.refine_batch_size),
                "--coarse_paths_per_segment",
                str(payload.coarse_paths_per_segment),
                "--coarse_margin",
                str(payload.coarse_margin),
                "--refine_margin",
                str(payload.refine_margin),
                "--working_resolution",
                str(payload.working_resolution),
                "--coarse_compactness",
                str(payload.coarse_compactness),
                "--refine_compactness",
                str(payload.refine_compactness),
                "--slic_sigma",
                str(payload.slic_sigma),
                "--learning_rate",
                str(payload.learning_rate),
                "--path_penalty",
                str(payload.path_penalty),
                "--seed",
                str(payload.seed),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "MPLCONFIGDIR": "/tmp/supersvg-matplotlib"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if not active_job.attach(job_id, process):
                yield json.dumps({"type": "cancelled", "message": "Superseded by a newer request."}) + "\n"
                return

            recent_lines = []
            try:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    recent_lines.append(line)
                    recent_lines = recent_lines[-20:]
                    if line.startswith("SUPERSVG_PROGRESS "):
                        _, percent, message = line.split(" ", 2)
                        event = {"type": "progress", "percent": int(percent), "message": message}
                    elif line.startswith("SUPERSVG_PREVIEW "):
                        preview_path = Path(line.split(" ", 1)[1])
                        if preview_path.exists():
                            event = {"type": "preview", "svg": preview_path.read_text(encoding="utf-8")}
                        else:
                            event = {"type": "log", "message": "Initial SVG preview was not found."}
                    elif line.startswith("SUPERSVG_DIAGNOSTIC "):
                        _, name, artifact = line.split(" ", 2)
                        artifact_path = Path(artifact)
                        if artifact_path.suffix.lower() == ".svg":
                            event = {"type": "diagnostic", "name": name, "kind": "svg", "data": artifact_path.read_text(encoding="utf-8")}
                        else:
                            encoded_artifact = base64.b64encode(artifact_path.read_bytes()).decode("ascii")
                            event = {"type": "diagnostic", "name": name, "kind": "image", "data": f"data:image/png;base64,{encoded_artifact}"}
                    else:
                        event = {"type": "log", "message": line}
                    yield json.dumps(event) + "\n"
            finally:
                if process.poll() is None:
                    active_job.terminate(process)

            return_code = process.wait()
            is_current = active_job.is_current(job_id)
            active_job.finish(job_id, process)
            if not is_current:
                yield json.dumps({"type": "cancelled", "message": "Cancelled by a newer request."}) + "\n"
                return
            if return_code != 0:
                detail = "\n".join(recent_lines[-12:]) or "Vectorization failed."
                yield json.dumps({"type": "error", "message": detail}) + "\n"
                return

            svg_path = output_dir / "input.svg"
            if not svg_path.exists():
                yield json.dumps({"type": "error", "message": "Vectorization completed without producing an SVG."}) + "\n"
                return

            svg = svg_path.read_text(encoding="utf-8")
            yield json.dumps({
                "type": "result",
                "svg": svg,
                "filename": "supersvg-output.svg",
                "settings": {
                    **json.loads(payload.model_dump_json(exclude={"image"})),
                    "device": device,
                },
            }) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    reload_enabled = os.environ.get("SUPERSVG_RELOAD", "1").lower() in {"1", "true", "yes"}
    options = {
        "host": os.environ.get("HOST", "0.0.0.0"),
        "port": int(os.environ.get("PORT", "8000")),
        "reload": reload_enabled,
    }
    if reload_enabled:
        options.update(
            reload_dirs=[str(ROOT)],
            reload_excludes=[
                ".git/*",
                "__pycache__/*",
                "output/*",
                "output_ratio_check/*",
                "test_matrix/*",
            ],
        )
    uvicorn.run("server:app" if reload_enabled else app, **options)
