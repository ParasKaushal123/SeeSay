import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_dataset
from .demo import make_demo
from .training import EvidenceModel, TrainConfig, train_to_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight CXR evidence models from cached features")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fit probes, LightGBM, and held-out sigmoid calibration")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--folds", type=int, default=3)
    training.add_argument("--trees", type=int, default=80)
    training.add_argument("--threads", type=int, default=2)
    training.add_argument("--seed", type=int, default=42)
    demo = commands.add_parser("demo", help="Run a synthetic fixture, not a medical benchmark")
    demo.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict", help="Score cached features using a trusted local model artifact")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            if args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {args.output}")
            source = make_demo(args.output / "synthetic_features.npz")
            report = train_to_directory(source, args.output / "run")
        elif args.command == "train":
            config = TrainConfig(folds=args.folds, seed=args.seed, trees=args.trees, threads=args.threads)
            report = train_to_directory(args.data, args.output, config)
        else:
            if args.output.exists():
                raise FileExistsError(args.output)
            if args.output.suffix != ".npz":
                raise ValueError("Prediction output must have the .npz extension")
            # joblib is pickle-based: never load artifacts from untrusted sources.
            model = joblib.load(args.model)
            if not isinstance(model, EvidenceModel) or model.schema_version != 1:
                raise ValueError("Unsupported model artifact")
            data = load_dataset(args.data, require_labels=False)
            probabilities = model.predict(data)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, study_ids=data.study_ids,
                                finding_names=np.array(model.finding_names), probabilities=probabilities,
                                calibration_label_source=np.array(model.label_source))
            print(json.dumps({"studies": len(data.study_ids), "output": str(args.output)}))
            return
        print(json.dumps({"label_source": report["label_source"],
                          "training_seconds": report["training_seconds"],
                          "output": str(args.output)}, indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
