import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "experiment_name", help="The name of the experiment"
    )
    parser.add_argument("--rps", nargs="+", type=int, default = [800], help="The number of requests per second")
    parser.add_argument(
        "--starting_sequence", type=int, default=0, help="The starting sequence in case the sequence is started in the middle."
    )
    parser.add_argument(
        "--config_file", help="The file containing all the configurations of the experiment."
    )
    parser.add_argument(
        "--overwrite_experiment_folder",
        action="store_true",
        help="If set, overwrites the experiment results from a previous run.",
    )

    parser.add_argument("--bottlenecked_nodes", nargs="+", type=str, help="The nodes that need to be bottlenecked.")
    parser.add_argument("--interference_percentage", nargs="+", type=int, help="The interference measure (e.g., percentage) over each bottleneck period.")
    parser.add_argument("--phases", nargs="+", type=int, help="The periods of non-bottlenecked and bottlenecked duration in seconds.")

    return parser.parse_args()