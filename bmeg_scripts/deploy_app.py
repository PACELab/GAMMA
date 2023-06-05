import os
import yaml

class Deployment:
    def __init__(self, args):
        self.parameters_file = args.parameters_file
        self.k8s_folder = args.k8s_folder
    
    def create_manifest_files(self):
        pass

    def write_manifest_file(self, manifest_file, data=""):
        """
        Write a manifest file. If data is not provided, creates an empty file.
        """
        with open(os.path.join(self.k8s_folder, manifest_file), "w") as f:
            yaml.dump_all(data, f, default_flow_style=False)

    def load_multi_doc_yaml(self, manifest_file):
        """
        Load a manifest file.
        """
        with open(manifest_file) as f:
            data = list(yaml.load_all(f, Loader=yaml.FullLoader))
    
        return data
    





    