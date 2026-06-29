from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def white_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (255, 255, 255))


def main() -> None:
    parser = argparse.ArgumentParser("Combine two qualitative panels side by side.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "generated_figures" / "paper_qualitative_panels" / "fig_qata_mosmed_combined.png")
    parser.add_argument("--gap", type=int, default=28)
    parser.add_argument("--margin", type=int, default=20)
    args = parser.parse_args()

    left_path = args.left if args.left.is_absolute() else ROOT / args.left
    right_path = args.right if args.right.is_absolute() else ROOT / args.right
    out_path = args.out if args.out.is_absolute() else ROOT / args.out

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = max(left.height, right.height)
    width = args.margin * 2 + left.width + args.gap + right.width
    out = white_canvas(width, height + args.margin * 2)
    out.paste(left, (args.margin, args.margin))
    out.paste(right, (args.margin + left.width + args.gap, args.margin))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    out.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
