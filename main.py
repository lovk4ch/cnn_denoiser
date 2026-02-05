import shutil
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from dncnn import Denoiser
from noise import add_noise
from utils import tensor_to_jpg, last_file_index

TRAIN = True
CLEAN_CACHE = True

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ImageDataset(
    root="data/src",
    transform=transform,
    crop_size=[128, 256, 384],
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    drop_last=False,
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = Path("models")
    res_path = Path("data/res")
    train_path = Path("data/train")

    if CLEAN_CACHE:
        if train_path.exists():
            shutil.rmtree(train_path)

    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "denoiser.pth"

    res_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)

    model = Denoiser()
    if model_file.exists():
        model.load_state_dict(torch.load(model_file, map_location=device))
        print("✔️️ Model weights loaded from disk.")
    else:
        print("⚠️ Model config doesn't exist, start training from scratch.")
    model.to(device)

    criterion = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    if TRAIN:
        last_epoch = last_file_index(res_path)
        index = 1
        model.train()

        for epoch in range(last_epoch, 500):
            train_bar = tqdm(train_loader, desc=f"\033[94m{index}")

            for batch in train_bar:
                batch = batch.to(device)
                noisy = add_noise(batch)
                batch = (batch * 2 - 1)
                noisy = (noisy * 2 - 1)

                base_noise = noisy - batch
                res_noise = model(noisy)

                # сумма квадратов разностей между пикселями картинок
                loss = criterion(base_noise, res_noise)

                # --- baseline: насколько noisy хуже clean ---
                with torch.no_grad():
                    mse_noisy = criterion(noisy, batch).item()

                # --- основной loss ---
                mse_pred = loss.item()

                # --- gain: во сколько раз модель лучше noisy ---
                gain = mse_noisy / (mse_pred + 1e-8)

                optim.zero_grad()
                loss.backward()

                # --- градиенты ---
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.detach().pow(2).sum().item()
                grad_norm = grad_norm ** 0.5

                optim.step()

                train_bar.set_postfix(
                    mse_pred=f"{mse_pred:.4f}",
                    gain=f"{gain:.2f}",
                    grad_norm=f"{grad_norm:.3e}",
                    iter=f"{index}"
                )

                # '''
                if index % 1 == 0:
                    step_res = torch.cat([batch[0], noisy[0], (res_noise - base_noise)[0]], dim=2)
                    tensor_to_jpg(step_res, f"data/train/step_res_{index}.jpg")
                # '''

                index += 1

            evaluate(model, device, epoch)
            model.train()

            torch.save(model.state_dict(), model_file)
            print("Model state saved successfully")

    else:
        evaluate(model, device)


def evaluate(model, device, epoch=1):
    image = Image.open("test.jpg").convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    image = image * 2 - 1

    model.eval()
    with torch.no_grad():
        out = model(image)
        res = (image - out).clamp(-1, 1)
        tensor_to_jpg(res, f"data/res/res_ep_{epoch + 1}.jpg")


if __name__ == "__main__":
    main()