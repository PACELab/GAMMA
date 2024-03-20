import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "experiment_name", help="The name of the experiment"
    )
    parser.add_argument("--rps", nargs="+", type=int, default = [800], help="The number of requests per second")
    parser.add_argument("--bottleneck_type", default = "cpu", choices =["cpu", "memory", "io", "network"],  help="The type of bottleneck")
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
    parser.add_argument("--experiment_duration", type=int, default = 1200, help="The experiment duration (excluding warm up) in seconds.")
    parser.add_argument("--warm_up_duration", type=int, default = 300, help="The warm up duration in seconds.")
    
    parser.add_argument("--cpu_bottlenecked_nodes", nargs="+", type=str, help="The nodes that need to be cpu bottlenecked.")
    parser.add_argument("--cpu_interference_percentage", nargs="+", type=int, help="The interference measure (e.g., percentage) over each bottleneck period.")
    parser.add_argument("--cpu_phases", nargs="+", type=int, help="The periods of non-bottlenecked and bottlenecked duration in seconds.")
       
    parser.add_argument("--mem_bottlenecked_nodes", nargs="+", type=str, help="The nodes that need to be memory bottlenecked.")
    parser.add_argument("--mem_interference_percentage", nargs="+", type=int, help="The interference measure (e.g., percentage) over each bottleneck period.")
    parser.add_argument("--mem_phases", nargs="+", type=int, help="The periods of non-bottlenecked and bottlenecked duration in seconds.")    

    parser.add_argument("--net_bottlenecked_nodes", nargs="+", type=str, help="The nodes that need to be cpu bottlenecked.")
    parser.add_argument("--net_interference_percentage", nargs="+", type=int, help="The interference measure (e.g., percentage) over each bottleneck period.")
    parser.add_argument("--net_phases", nargs="+", type=int, help="The periods of non-bottlenecked and bottlenecked duration in seconds.")
    
    parser.add_argument("--io_bottlenecked_nodes", nargs="+", type=str, help="The nodes that need to be cpu bottlenecked.")
    parser.add_argument("--io_interference_percentage", nargs="+", type=int, help="The interference measure (e.g., percentage) over each bottleneck period.")
    parser.add_argument("--io_phases", nargs="+", type=int, help="The periods of non-bottlenecked and bottlenecked duration in seconds.")
    
    parser.add_argument('--skip_trace_collection', action='store_true', help="Skips Jaeger trace collection.")
    parser.add_argument('--skip_log_metric_collection', action='store_true', help="Skips log and prom metric collection.")
    return parser.parse_args()