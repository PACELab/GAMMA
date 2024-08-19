import os
import logging
import pickle
import torch
import numpy as np
import random
import json
import logging
import dgl
import inspect

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        logging.raiseExceptions("File path "+filepath+" not exists!")
        return

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def get_metrics(predictions, true_labels):
    assert len(predictions) == len(true_labels), "Length mismatch between predictions and true_labels."

    TP, TN, FP, FN = 0, 0, 0, 0
    
    for pred, true in zip(predictions, true_labels):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1_score


def collate(samples):
    # Assuming samples is a list of tuples, where each tuple contains a graph and a label.
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batch_size = len(graphs)
    # Extract the features from the batched graph.
    # Assuming 'features' is the key where node features are stored.
    lat = batched_graph.ndata['latency']    
    input_lat = lat.shape[0] // batch_size
    reshaped_lat = lat.view(batch_size, input_lat, -1).permute(0, 2, 1)

    cpu = batched_graph.ndata['container_cpu_usage_seconds_total']
    input_cpu = cpu.shape[0] // batch_size
    reshaped_cpu = cpu.view(batch_size, input_cpu, -1).permute(0, 2, 1)

    mem = batched_graph.ndata['container_memory_usage_bytes']
    input_mem = mem.shape[0] // batch_size
    reshaped_mem = mem.view(batch_size, input_mem, -1).permute(0, 2, 1)

    nout = batched_graph.ndata['container_network_transmit_bytes_total']
    input_nout = nout.shape[0] // batch_size
    reshaped_nout = nout.view(batch_size, input_nout, -1).permute(0, 2, 1)

    nin = batched_graph.ndata['container_network_receive_bytes_total']
    input_nin = nin.shape[0] // batch_size
    reshaped_nin = nin.view(batch_size, input_nin, -1).permute(0, 2, 1)
    # Reshape the features from (batch_size*input_dim, T) to (batch_size, T, input_dim)

    return batched_graph, reshaped_lat, reshaped_lat, reshaped_mem, reshaped_nout, reshaped_nin, torch.tensor(labels)


def save_logits_as_dict(logits, keys, filename):
    """
    Saves a list of tensors as a dictionary with variable names as keys and tensor values as dictionary values.
    """
    # Get the previous frame (caller frame)
    frame = inspect.currentframe().f_back
    tensor_dict = {}
    
    # Loop through the tensors
    for logit in logits:
        # Find all variable names that this tensor object is assigned to
        names = [name for name, var in frame.f_locals.items() if torch.is_tensor(var) and var is tensor and not name.startswith('_')]
        
        # If there's a name that refers to this tensor, add to dictionary
        if names:
            tensor_dict[names[0]] = logit
            
    return tensor_dict

