import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev

from tqdm import tqdm


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m compute_stats",
        description="Compute Mean and Standard Deviations of the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-folder",
        metavar="DATA_FOLDER",
        type=str,
        default="output",
        help="Path to the folder containing the data.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="output",
        help="Path to the output directory.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args: Namespace):
    """
    Main function to compute the mean and standard deviations of the results.

    Parameters
    ----------
    args: Namespace
        Parsed arguments passed through command line.
    """

    results = {}
    for task in tqdm(os.listdir(args.data_folder)):
        task_folder = Path(args.data_folder) / task
        if not os.path.isdir(task_folder):
            continue
        if task not in results:
            results[task] = {}

        for model in tqdm(os.listdir(task_folder), leave=False):
            if model not in results[task]:
                results[task][model] = {}

            for experiment_name in os.listdir(task_folder / model):
                if os.path.isdir(
                    Path(args.data_folder) / task / model / experiment_name
                ):
                    if experiment_name.split("_")[0] in [
                        "none",
                        "full",
                        "gold",
                        "retrieved",
                    ]:
                        knowledge = "_".join(experiment_name.split("_")[:-1])
                        if knowledge not in results[task][model]:
                            results[task][model][knowledge] = {
                                "nlls": [],
                                "ppls": [],
                            }
                        # compute fine-tuning
                        with open(
                            Path(args.data_folder)
                            / task
                            / model
                            / experiment_name
                            / "fine_tune.txt",
                            "r",
                        ) as f:
                            prompting_res = f.readlines()
                            nll, ppl = (
                                prompting_res[0].strip().split()[-1],
                                prompting_res[1].strip().split()[-1],
                            )
                            results[task][model][knowledge]["nlls"].append(float(nll))
                            results[task][model][knowledge]["ppls"].append(float(ppl))

                        if knowledge not in results[task][model]:
                            results[task][model][knowledge] = {}

            for knowledge in results[task][model]:
                nlls = results[task][model][knowledge]["nlls"]
                ppls = results[task][model][knowledge]["ppls"]

                results[task][model][knowledge] = {
                    "mean_nll": mean(nlls),
                    "std_nll": stdev(nlls),
                    "mean_ppl": mean(ppls),
                    "std_ppl": stdev(ppls),
                }

    # write results to file
    if len(results) > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(Path(args.out_dir) / "results.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
