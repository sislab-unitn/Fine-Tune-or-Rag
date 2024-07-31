import json
import sys
from pprint import pprint


def main(optimization_file):
    """
    Given an optimization file, this function prints the best parameters and the BLEU score.

    Parameters
    ----------
    optimization_file : str
        The path to the optimization file.
    """
    with open(optimization_file, "r") as f:
        data = json.load(f)

    results = [(results["bleu"], params) for params, results in data.items()]

    bleu, best_params = max(results)

    top_p, temperature, top_k = best_params.split("_")

    print("Best parameters:")
    pprint(
        {
            "top_p": top_p,
            "temperature": temperature,
            "top_k": top_k,
        },
        indent=4,
    )
    print(f"BLEU: {bleu}")


if __name__ == "__main__":

    assert len(sys.argv) == 2, "Please provide the optimization file as an argument."

    main(sys.argv[1])
