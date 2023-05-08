

import arugments
import deploy_app

def main():
    faulthandler.enable()
    args = arguments.argument_parser()
    deploy_app.Deployment(args)
