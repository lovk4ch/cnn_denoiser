from pathlib import Path

import torch
import yaml


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path=None):
    if path is None:
        base_dir = Path(__file__).resolve().parents[1]
        path = base_dir / "config" / "project.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def load_weights(model, cfg, device):
    path = Path(cfg["data"]["ckpt_dir"]) / cfg["data"]["denoise_path"]
    print(f"Loading {Path.cwd() / path}")

    if not path.exists():
        print(f"⚠️ Model {path.name} not found, training from scratch.")
        return False

    try:
        state = torch.load(
            path,
            weights_only=True,
            map_location=device
        )
        model.load_state_dict(state)
        print(f"✔️ Model {path.name} weights loaded.")
        return True

    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        print(f"⚠️ Training from scratch.")
        return False

def save_weights(model, cfg):
    path = Path(cfg["data"]["ckpt_dir"]) / cfg["data"]["denoise_path"]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"✔️ Model {path.name} checkpoint updated successfully.")
    except Exception as e:
        print(f"❌ Failed to save model {path.name} : {e}")