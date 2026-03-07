"""Command-line interface for FLENcognition.

Usage
-----
    python -m flencognition [OPTIONS] IMAGE [IMAGE ...]

Or, after installing the package::

    flencognition [OPTIONS] IMAGE [IMAGE ...]
"""

from __future__ import annotations

import argparse
import sys

from . import FLENcognition


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flencognition",
        description=(
            "FLENcognition – convert document images to Markdown using "
            "the FireRed-OCR model."
        ),
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="One or more image files to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="md_output",
        metavar="DIR",
        help="Directory for saved Markdown files (default: md_output).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save each result as a .md file in --output-dir.",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help='PyTorch device string, e.g. "cpu" or "cuda" (auto-detected by default).',
    )
    parser.add_argument(
        "--model-dir",
        default="FireRedTeam/FireRed-OCR",
        metavar="MODEL",
        help=(
            "Hugging Face repository or local path for the model "
            '(default: "FireRedTeam/FireRed-OCR").'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = FLENcognition(
        model_dir=args.model_dir,
        device=args.device,
        output_dir=args.output_dir,
    )

    exit_code = 0
    for image_path in args.images:
        try:
            result = engine.process_image(image_path, save_markdown=args.save)
            if args.save and result["file"]:
                print(f"✅  {image_path} → {result['file']}", file=sys.stderr)
            print(result["markdown"])
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"❌  Error processing {image_path}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
