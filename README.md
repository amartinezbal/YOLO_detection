# YOLO Detection

Fine-tune Ultralytics YOLO models, export them for deployment, and use the included mobile inference examples.

## Project Structure

- `scripts/`: training and model-export command-line tools.
- `mobile/ios/` and `mobile/react-native/`: platform integration examples. These are sample source files, not complete mobile app projects.
- `assets/labels/`: label lists for COCO and Objects365.
- `models/checkpoints/` and `models/exports/`: local model checkpoints and exported models. Model binaries are not included in Git.
- `data/`: local datasets and calibration inputs. Dataset files are not included in Git.
- `artifacts/training/runs/`: generated training and validation outputs. These are not included in Git.

## Setup

Requires Python 3.11.12 or newer. Install the core dependencies with `uv sync`. To enable all export formats, install the optional export dependencies with `uv sync --extra export`.

Run commands from the repository root. Fine-tune on a YOLO dataset YAML:

```sh
uv run python scripts/main.py --data /path/to/dataset.yaml --epochs 100 --export onnx
```

The default initial model is `yolov8n.pt`; Ultralytics downloads it if it is not available locally. Training output is written under `artifacts/training/runs/` by default. For a short pipeline check using COCO8:

```sh
uv run python scripts/main.py --smoke-test
```

Export an existing checkpoint:

```sh
uv run python scripts/export_only.py --weights models/checkpoints/best.pt --format tflite
```

For INT8 TFLite export, provide a local dataset YAML with `--data` and include `--int8`.

## Mobile Examples

The React Native sample expects `mobile/react-native/assets/yolov8n_int8.tflite`. Add the TFLite model there before integrating the component into a React Native app. The iOS example expects `yolov8n.mlpackage` to be added to the Xcode target. Exported models are intentionally kept out of the public source tree; generate or obtain a model compatible with the sample before use.
