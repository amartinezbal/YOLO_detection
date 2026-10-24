import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
EXPORT_FORMATS = ("tflite", "coreml", "onnx")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def dataset_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        raise argparse.ArgumentTypeError("must point to an existing .yaml or .yml file")
    return path


def export_model(
    weights: str | Path,
    output_format: str = "tflite",
    image_size: int = 640,
    int8: bool = False,
    data: str | Path | None = None,
) -> Path:
    if output_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {output_format}")
    if image_size < 1:
        raise ValueError("Image size must be positive")
    if int8 and (output_format != "tflite" or data is None):
        raise ValueError("INT8 requires TFLite and a calibration dataset")

    from ultralytics import YOLO

    if int8:
        from ultralytics.data.utils import check_det_dataset

        check_det_dataset(str(data), autodownload=False)

    model = YOLO(str(weights))
    options = {
        "format": output_format,
        "imgsz": image_size,
        "device": "cpu",
        "nms": output_format == "coreml",
    }
    if output_format == "tflite":
        options["int8"] = int8
        if int8:
            options["data"] = str(data)

    output = Path(model.export(**options))
    labels = output.parent / f"{output.stem}_labels.txt"
    labels.write_text(
        "".join(f"{model.names[index]}\n" for index in range(len(model.names))),
        encoding="utf-8",
    )
    LOGGER.info("Export saved to %s", output)
    LOGGER.info("Class labels saved to %s", labels)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint for mobile inference.")
    parser.add_argument("--weights", required=True, help="Checkpoint path or pretrained model name.")
    parser.add_argument("--format", choices=EXPORT_FORMATS, default="tflite")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--int8", action="store_true", help="Quantize TFLite using representative images.")
    parser.add_argument("--data", type=dataset_path, help="Local YAML for INT8 calibration.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.int8 and (args.format != "tflite" or args.data is None):
        parser.error("--int8 requires --format tflite and --data")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    export_model(args.weights, args.format, args.imgsz, args.int8, args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())