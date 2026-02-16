import base64
import io

from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from PIL import Image
import torch

from denoiser import Denoiser
from denoiser.utils.common import get_transform, tensor_to_jpg
from denoiser.utils.model import load_config, get_device, load_weights


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


@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    transform = get_transform(normalize=True)
    model = app.state.model
    image = transform(image).to(app.state.device)

    with torch.no_grad():
        out = (image - model(image)).clamp(-1, 1)

    out = tensor_to_jpg(out)
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    decoded_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded_bytes))
    image.show()

    return {"image": encoded}
