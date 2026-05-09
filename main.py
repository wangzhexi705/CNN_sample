import argparse

from src.config import TrainConfig, build_run_name
from src.gradcam import generate_gradcam
from src.train import evaluate_checkpoint, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="STL-10 image classification")
    parser.add_argument("--mode", choices=["train", "eval", "gradcam"], default="train")
    parser.add_argument("--model", choices=["basic", "augmented", "improved"], default="basic")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--pooling", choices=["max", "avg"], default="avg")
    parser.add_argument("--normalization", choices=["batchnorm", "none"], default="batchnorm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--class-name", choices=[
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ], default=None)
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = build_run_name(
        args.model,
        activation=args.activation,
        pooling=args.pooling,
        normalization=args.normalization,
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        device=args.device,
        use_augmentation=args.model == "augmented",
        activation=args.activation,
        pooling=args.pooling,
        normalization=args.normalization,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
        checkpoint_name=f"{run_name}_cnn_best.pt",
    )

    if args.mode == "train":
        train_model(config, model_name=args.model)
    elif args.mode == "eval":
        evaluate_checkpoint(config, model_name=args.model)
    elif args.mode == "gradcam":
        generate_gradcam(
            config,
            model_name=args.model,
            image_path=args.image_path,
            split=args.split,
            sample_index=args.sample_index,
            class_name=args.class_name,
            target_class=args.target_class,
            alpha=args.alpha,
            output_tag=args.output_tag,
        )


if __name__ == "__main__":
    main()
