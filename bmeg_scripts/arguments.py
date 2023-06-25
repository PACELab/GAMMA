import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "experiment_name", help="The name of the experiment"
    )
    parser.add_argument("--rps", type=int, default = 800, help="The number of requests per second")
    parser.add_argument(
        "--config_file", help="The file containing all the configurations of the experiment."
    )
    parser.add_argument(
        "--overwrite_experiment_folder",
        action="store_true",
        help="If set, overwrites the experiment results from a previous run.",
    )

    return parser.parse_args()