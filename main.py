from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.v2.functional import to_pil_image

from dataset import ImageDataset
from dncnn import Denoiser
from noise import add_canon_like_noise

TRAIN = True

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
    else:
        print("⚠️ Model config doesn't exist, start training from scratch.")
    model.to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    if TRAIN:
        index = 1
        model.train()
        for epoch in range(5):
            w_before = model.conv_in.weight.detach().clone()
            for batch in train_loader:
                batch = batch.to(device)
                noisy = add_canon_like_noise(batch)

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

                # --- градиенты ---
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.detach().pow(2).sum().item()
                grad_norm = grad_norm ** 0.5

                optim.zero_grad()
                loss.backward()
                optim.step()

                print(
                    f"mse_noisy={mse_noisy:.4f} | "
                    f"mse_pred={mse_pred:.4f} | "
                    f"gain={gain:.2f} | "
                    f"grad_norm={grad_norm:.3e}"
                )

                # print(f"epoch={epoch}, loss={loss.item():.6f}")

                to_pil_image(noisy.detach().clamp(0, 1).squeeze(0).cpu()).save(f"data/noisy/{index}_2_noisy.jpg")
                to_pil_image(batch.detach().clamp(0, 1).squeeze(0).cpu()).save(f"data/noisy/{index}_1_gt.jpg")

                n = res_noise.detach().cpu().squeeze(0)
                scale = n.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
                n = (n / scale) * 0.5 + 0.5
                to_pil_image(n).save(f"data/noisy/{index}_3_res_noise.jpg")

                e = (res_noise - base_noise).detach().abs().cpu().squeeze(0)  # [3,H,W]
                e = e / (e.amax(dim=(1, 2), keepdim=True) + 1e-8)
                to_pil_image(e).save(f"data/noisy/{index}_4_loss.jpg")

                index += 1

            dw = (model.conv_in.weight.detach() - w_before).abs().mean().item()
            print("mean |Δw| conv_in: ", dw)

            torch.save(model.state_dict(), model_file)
            print("Model state saved successfully")

    else:
        image = Image.open("data/test.jpg").convert('RGB')
        noisy = transform(image).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            out = model(noisy.to(device)).clamp(0.0, 1.0)
            res = torch.cat([noisy, out, (noisy - out)], dim=3).squeeze(0).cpu()
            to_pil_image(res).show()

if __name__ == "__main__":
    main()