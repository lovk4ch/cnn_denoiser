import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from denoiser import Denoiser
from denoiser.utils.common import get_transform, tensor_to_jpg
from denoiser.utils.model import load_config, get_device, load_weights

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_config()
    device = get_device()
    model = Denoiser(cfg["denoise"]).to(device)
    model.eval()

    load_weights(model, cfg, device)

    _app.state.device = device
    _app.state.model = model
    yield

    del _app.state.model
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    transform = get_transform(normalize=True)
    model = app.state.model
    image = transform(image).to(app.state.device)

    with torch.no_grad():
        image = (image - model(image)).clamp(-1, 1)
        image = image.detach()
        out = (image - model(image)).clamp(-1, 1)

    out = tensor_to_jpg(out)

    buffer = io.BytesIO()
    out.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/jpg",
        headers={
            "Content-Disposition": f"attachment; filename=denoised_{file.filename}"
        }
    )
