from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from dncnn import Denoiser
from noise import add_canon_like_noise, add_gaussian_noise
from utils import tensor_to_jpg

TRAIN = False

model_path = Path("models/")
model_path.mkdir(parents=True, exist_ok=True)
model_file = model_path / "denoiser.pth"

out_path = Path("data/noisy")
out_path.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ImageDataset(
    root="data/train",
    transform=transform
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

    model = Denoiser()
    if model_file.exists():
        model.load_state_dict(torch.load(model_file, map_location=device))
        print("✔️️ Model weights loaded from disk.")
    else:
        print("⚠️ Model config doesn't exist, start training from scratch.")
    model.to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=4e-4)

    if TRAIN:
        index = 1
        model.train()
        noise = torch.zeros(1, 3, 512, 512)
        noise = add_gaussian_noise(noise)

        for epoch in range(15):
            for batch in train_loader:
                batch = batch.to(device)
                noisy = (batch + noise).clamp(-1, 1)

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

                print("res_noise abs mean:", res_noise.abs().mean().item(),
                      "base_noise abs mean:", base_noise.abs().mean().item())

                print(
                    f"mse_noisy={mse_noisy:.4f} | "
                    f"mse_pred={mse_pred:.4f} | "
                    f"gain={gain:.2f} | "
                    f"grad_norm={grad_norm:.3e}"
                )

                # print(f"epoch={epoch}, loss={loss.item():.6f}")

                if index == 1:
                    tensor_to_jpg(batch, f"basic.jpg")
                    tensor_to_jpg(noisy, f"test.jpg")
                tensor_to_jpg(res_noise, f"data/noisy/res_noise_{index}.jpg", normalize=True)
                tensor_to_jpg(res_noise - base_noise, f"data/noisy/res_loss_{index}.jpg", normalize=True)

                index += 1

            torch.save(model.state_dict(), model_file)
            print("Model state saved successfully")

    else:
        image = Image.open("test.jpg").convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        image = image * 2 - 1

        basic = Image.open("basic.jpg").convert('RGB')
        basic = transform(basic).unsqueeze(0).to(device)
        basic = basic * 2 - 1

        model.eval()
        with torch.no_grad():
            out = model(image)
            res = (image - out).clamp(-1, 1)
            res = torch.cat([basic, image, res], dim=3)
            tensor_to_jpg(res, "res.jpg")

if __name__ == "__main__":
    main()