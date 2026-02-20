import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from denoiser import Denoiser
from denoiser.utils.common import img_to_tensor, tensor_to_img
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
app.mount("/static", StaticFiles(
    directory=FRONTEND_DIR),
    name="static"
)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile, iterations: int = Form(...)):
    image = Image.open(file.file).convert("RGB")
    transform = img_to_tensor(normalize=True, max_size=460)
    model = app.state.model
    image = transform(image).to(app.state.device)

    with torch.no_grad():
        for i in range(iterations):
            image = (image - model(image)).clamp(-1, 1)
            image = image.detach()

    image = tensor_to_img(image)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")