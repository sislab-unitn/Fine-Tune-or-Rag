import argparse
from argparse import Namespace


import torch


from subparsers import baseline, fine_tune, generate, int_grad, optimize, prompting


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model",
        metavar="MODEL_NAME",
        choices=["llama", "mistral", "gpt2"],
        help="Decide which model to use.",
    )
    parser.add_argument(
        "data_folder",
        metavar="DATA_FOLDER",
        type=str,
        help="Path to the folder containing the data.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="output",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split the model across multiple GPUs.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--unstr-know",
        metavar="TYPE",
        type=str,
        choices=["none", "full", "retrieved", "gold"],
        default="none",
        help="Unstructured Knowledge provided to the model.",
    )
    parser.add_argument(
        "--top-k",
        metavar="TOP_K",
        type=int,
        default=5,
        help="Top k documents to retrieve.",
    )

    subparsers = parser.add_subparsers(help="", dest="command")
    baseline.configure_subparsers(subparsers)
    fine_tune.configure_subparsers(subparsers)
    int_grad.configure_subparsers(subparsers)
    generate.configure_subparsers(subparsers)
    optimize.configure_subparsers(subparsers)
    prompting.configure_subparsers(subparsers)

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    args = get_args()
    args.func(args)
