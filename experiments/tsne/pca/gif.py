from pathlib import Path
from PIL import Image
import argparse


def make_gif(source_dir: Path, pattern: str, output: Path, duration: int) -> None:
    frames = sorted(source_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No files matching '{pattern}' in {source_dir}")
    images = [Image.open(frame).convert("RGB") for frame in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GIF from *_cls0.5.png files.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory to scan")
    parser.add_argument("--output", type=Path, default=Path("animation.gif"), help="GIF file path")
    parser.add_argument("--pattern", type=str, default="*_cls0.5.png", help="File pattern to match")
    parser.add_argument("--duration", type=int, default=200, help="Frame duration in ms")
    args = parser.parse_args()

    make_gif(args.dir, args.pattern, args.output, args.duration)


if __name__ == "__main__":
    main()