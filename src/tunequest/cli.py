"""Command line entry points; ML imports are deferred until needed."""
import argparse
import json
from pathlib import Path
from .common import binary_metrics, best_threshold


def read_training_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "model_id", "dataset_path", "output_dir", "max_seq_length", "seed", "num_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size",
        "learning_rate", "weight_decay", "warmup_steps", "lora_r", "lora_alpha", "lora_dropout",
    }
    if set(config) != required:
        raise ValueError(f"Config keys differ: missing={sorted(required-set(config))}, unknown={sorted(set(config)-required)}")
    for name in ("max_seq_length", "num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size", "lora_r", "lora_alpha"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["warmup_steps"] < 0 or not 0 < config["learning_rate"] < 1 or not 0 <= config["lora_dropout"] < 1 or config["weight_decay"] < 0:
        raise ValueError("Invalid training hyperparameters.")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="TuneQuest code clone research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preprocess", help="Create an audited HF DatasetDict")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--sample-size", type=int)
    prep.add_argument("--validation-fraction", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--symmetry", action="store_true")
    train = commands.add_parser("train", help="Run the preserved BF16 LoRA method")
    train.add_argument("--config", default="configs/bf16.json")
    train.add_argument("--check-config", action="store_true")
    inference = commands.add_parser("predict", help="Reconstructed adapter inference")
    inference.add_argument("--input", required=True)
    inference.add_argument("--adapter", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--max-length", type=int, default=1024)
    inference.add_argument("--batch-size", type=int, default=8)
    inference.add_argument("--tta", action="store_true")
    inference.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sub = commands.add_parser("submission", help="Threshold a p_final CSV")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--threshold", required=True, type=float)
    evaluate = commands.add_parser("evaluate", help="Evaluate existing NPY probabilities and labels")
    evaluate.add_argument("--probabilities", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "preprocess":
            from .preprocessing import preprocess
            result = preprocess(args.input, args.output, sample_size=args.sample_size,
                                validation_fraction=args.validation_fraction, seed=args.seed, symmetry=args.symmetry)
        elif args.command == "train":
            config = read_training_config(args.config)
            if args.check_config:
                result = config
            else:
                if not Path(config["dataset_path"]).is_dir():
                    raise FileNotFoundError(f"Missing processed dataset: {config['dataset_path']}")
                for path in (config["output_dir"], config["output_dir"] + "_final"):
                    if Path(path).exists():
                        raise FileExistsError(f"Refusing to overwrite training output: {path}")
                from .training import train
                train(config)
                result = {"adapter": config["output_dir"] + "_final"}
        elif args.command == "predict":
            from .prediction import predict
            result = predict(args.input, args.adapter, args.output, max_length=args.max_length,
                             batch_size=args.batch_size, tta=args.tta, device=args.device)
        elif args.command == "submission":
            from .prediction import submission
            result = submission(args.input, args.output, args.threshold)
        else:
            import numpy as np
            labels = np.load(args.labels, allow_pickle=False)
            probabilities = np.load(args.probabilities, allow_pickle=False)
            if labels.ndim != 1 or probabilities.ndim != 1:
                raise ValueError("Expected one-dimensional label and probability arrays.")
            result = best_threshold(labels, probabilities) if args.sweep else binary_metrics(labels, probabilities, args.threshold)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
