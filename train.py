import argparse

from denoiser import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval", "train"],
        default = "train",
        help="Which mode to run"
    )
    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer()
        trainer.run()
    else:
        exit()