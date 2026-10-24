import argparse
import logging
import math
from pathlib import Path

from export_only import EXPORT_FORMATS, dataset_path, export_model, positive_int

LOGGER = logging.getLogger(__name__)


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on a local dataset and optionally export it.")
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--data", type=dataset_path, help="Path to a local YOLO dataset YAML.")
    dataset.add_argument("--smoke-test", action="store_true", help="Download COCO8 and run a three-epoch pipeline check.")
    parser.add_argument("--model", default="yolov8n.pt", help="Initial checkpoint or pretrained model name.")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--batch", type=positive_int, default=8)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--workers", type=nonnegative_int, default=4)
    parser.add_argument("--device", help="Training device, such as 0, cpu, or mps; autodetected by default.")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="fine_tune")
    parser.add_argument("--lr0", type=positive_float, default=0.0001)
    parser.add_argument("--patience", type=positive_int, default=20)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--export", nargs="+", choices=EXPORT_FORMATS, default=[])
    parser.add_argument("--int8", action="store_true", help="Use INT8 for the requested TFLite export.")
    parser.add_argument("--webcam", action="store_true", help="Open camera 0 after validation and any exports.")
    return parser


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    data = "coco8.yaml" if args.smoke_test else str(args.data)
    check_det_dataset(data, autodownload=args.smoke_test)
    device = args.device if args.device is not None else select_device()
    if device == "cpu":
        LOGGER.warning("Training on CPU can be slow; use --device to select an accelerator.")
    LOGGER.info("Training %s on %s using %s", args.model, data, device)
    if args.smoke_test:
        LOGGER.warning("COCO8 is a pipeline check, not a meaningful accuracy benchmark.")

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=3 if args.smoke_test else args.epochs,
        imgsz=args.imgsz,
        batch=4 if args.smoke_test else args.batch,
        workers=0 if args.smoke_test else args.workers,
        device=device,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=False,
        optimizer="AdamW",
        lr0=args.lr0,
        patience=args.patience,
        seed=args.seed,
        cache=False,
        save=True,
        plots=True,
    )
    best_weights = Path(model.trainer.best).resolve()
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce a best checkpoint: {best_weights}")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=data, imgsz=args.imgsz, device=device)
    LOGGER.info("Validation mAP50: %.3f; mAP50-95: %.3f", metrics.box.map50, metrics.box.map)
    LOGGER.info("Best checkpoint: %s", best_weights)

    for output_format in dict.fromkeys(args.export):
        export_model(best_weights, output_format, args.imgsz, args.int8 and output_format == "tflite", data)

    if args.webcam:
        for _result in best_model.predict(source=0, show=True, stream=True, conf=0.45, device=device):
            pass
    return best_weights


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.int8 and "tflite" not in args.export:
        parser.error("--int8 requires --export tflite")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
