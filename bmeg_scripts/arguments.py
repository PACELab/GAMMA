import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--server_url", default="http://127.0.0.1:8000", help="The URL at which the SelfTune server is running"
    )
    return parser.parse_args()