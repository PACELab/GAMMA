from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import *
import pickle
import sys
import logging
from base import BaseModel
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class chunkDataset(Dataset): #[node_num, T, else]
    def __init__(self, chunks, edges):
        self.data = []
        self.idx2id = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["window_id"]
            self.idx2id[idx] = chunk_id
            graph = dgl.graph(edges)
            # print(graph.num_nodes())
            graph.ndata["latency"] = torch.FloatTensor(chunk["latency"].T.values)
            graph.ndata["container_cpu_usage_seconds_total"] = torch.FloatTensor(chunk["container_cpu_usage_seconds_total"].T.values)
            graph.ndata["container_network_transmit_bytes_total"] = torch.FloatTensor(chunk["container_network_transmit_bytes_total"].T.values)
            graph.ndata["container_memory_usage_bytes"] = torch.FloatTensor(chunk["container_memory_usage_bytes"].T.values)
            graph.ndata["container_network_receive_bytes_total"] = torch.FloatTensor(chunk["container_network_receive_bytes_total"].T.values)
            self.data.append((graph, [chunk["label_window"], chunk["label_window_238"], chunk["label_window_47"], 
                                      chunk["label_window_28"], chunk["label_window_247"], chunk["label_window_378"]]))
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]

def run(params):

    keys = ["window_pred", "window_true", "l1_pred", "l1_true", "l2_pred", "l2_true", 
            "l3_pred", "l3_true", "l4_pred", "l4_true", "l5_pred", "l5_true"]

    train_file = params["train_file"]
    with open(train_file, 'rb') as tr:
        train_chunks = pickle.load(tr)
    tr.close()

    test_file = params["test_file"]
    with open(test_file, 'rb') as te:
        test_chunks = pickle.load(te)
    te.close()


    metadata = read_json(params["metadata_json"])
    source = [float(x) for x in metadata["source"]]
    target = [float(x) for x in metadata["target"]]

    # print(source)
    # print(target)

    # edges = torch.stack((source, target), dim=0)
    edges = (source, target)
    train_data = chunkDataset(train_chunks, edges)
    test_data = chunkDataset(test_chunks, edges)

    train_dl = DataLoader(train_data, batch_size = params['batch_size'], shuffle=True, collate_fn=collate, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size = params['batch_size'], shuffle=False, collate_fn=collate, pin_memory=True)
    # print("Data loaded successfully!")
    logging.info("Data loaded successfully!")

    device = get_device(params["check_device"])
    model = BaseModel(nodes, device, epoches = evaluation_epochs, lr = params["learning_rate"], **params)
    train_logit_list = model.fit(train_dl)
    model = BaseModel(nodes, device, epoches = evaluation_epochs, lr = params["learning_rate"], **params)
    test_logit_list = model.evaluate(test_dl)  



# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    nodes = 30
    batch_size = 32
    random_seed = 12345
    evaluation_epochs = 20
    learning_rate = 0.001
    model = "all"
    result_dir = "./results"
    window_size = 60

    train_file = "./util_data/window_features_g1_60/full_data.pkl"
    test_file = "./util_data/window_features_g1_60/test_data.pkl"
    metadata_json = "./util_data/window_features_g1_60/metadata.json"

    features = ["latency", "container_cpu_usage_seconds_total", "container_memory_usage_bytes", 
                "container_network_transmit_bytes_total", "container_network_receive_bytes_total"]


    params = {'nodes': nodes,
            'batch_size': batch_size,
            'train_file': train_file,
            'test_file': test_file,
            'metadata_json': metadata_json,
            'learning_rate': learning_rate, 
            'model': 'all',
            'check_device': "gpu",
            'input_dims': window_size,
            'model_save_dir': result_dir
    }     

    run(params)
    # print(device)
    # print(type(device))


