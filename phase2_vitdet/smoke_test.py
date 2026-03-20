import argparse
import torch

from phase2_vitdet.simple_vitdet_fpn import build_vitdet_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for ViTDet bridge")
    parser.add_argument("--model", default="vit_base_patch16_224", type=str)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--align-channels", default=2048, type=int)
    parser.add_argument("--out-channels", default=256, type=int)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    bridge = build_vitdet_bridge(
        model_name=args.model,
        out_channels=args.out_channels,
        align_channels=args.align_channels,
        freeze_backbone=not args.unfreeze_backbone,
        use_align=True,
    ).to(args.device)

    dummy = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)
    with torch.no_grad():
        aligned, raw = bridge(dummy)

    print("Input:", dummy.shape)
    for name, feat in raw.items():
        print(f"raw[{name}]: {tuple(feat.shape)}")
    for name, feat in aligned.items():
        print(f"aligned[{name}]: {tuple(feat.shape)}")


if __name__ == "__main__":
    main()

