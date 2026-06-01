#!/usr/bin/env python3

from pathlib import Path
import argparse

import smollnet


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "abalone.data"


def load_abalone_dataset(path: Path):
    options = smollnet.CSVLoaderOptions()
    options.has_header = False
    options.target_columns = 1
    options.categorical_columns = [0]
    options.device = smollnet.Device.CUDA
    return smollnet.load_csv_dataset(str(path), options)


def build_network(input_features: int, output_features: int):
    return smollnet.Dense(
        smollnet.Linear(input_features, 64),
        smollnet.GeLU(),
        smollnet.Linear(64, 32),
        smollnet.GeLU(),
        smollnet.Linear(32, output_features),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a SmollNet regression model on UCI Abalone data."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=default_dataset_path(),
        help="Path to abalone.data",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    smollnet.manual_seed(args.seed)

    dataset = load_abalone_dataset(args.dataset)

    loader_options = smollnet.DataLoaderOptions()
    loader_options.batch_size = args.batch_size
    loader_options.shuffle = True
    loader_options.seed = args.seed

    input_features = dataset.inputs().size(1)
    output_features = dataset.targets().size(1)

    network = build_network(input_features, output_features)
    optimizer = smollnet.sgd(network.parameters(), lr=args.lr)
    loader = smollnet.DataLoader(dataset, loader_options)

    for epoch in range(args.epochs):
        total_loss = 0.0
        batches = 0

        for batch in loader:
            prediction = network.forward(batch.inputs)
            loss = smollnet.mse(prediction, batch.targets)
            total_loss += loss.item()
            batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / batches
        print(f"epoch[{epoch}] mean_loss={mean_loss:.4f}")

    sample = dataset.batch(0, 1)
    prediction = network.forward(sample.inputs)
    print(
        "sample "
        f"predicted_rings={prediction.item():.2f} "
        f"actual_rings={sample.targets.item():.2f}"
    )


if __name__ == "__main__":
    main()
