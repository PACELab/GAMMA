import os
import sys
import logging
from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import *
import pickle
import torch.nn as nn
import math
import pandas as pd
import os
import sys
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling
import math
import numpy as np
from utils import *
import time
import copy


os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=3, dev="cpu"):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), 
                       nn.BatchNorm1d(out_channels), nn.ReLU(), Chomp1d(padding)]
            
        self.network = nn.Sequential(*layers)
        
        self.out_dim = num_channels[-1]
        self.network.to(dev)
        
    
    def forward(self, x): #[batch_size, T, in_dim]

        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]

        return out


class SelfAttention(nn.Module):

    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):

        input_tensor = x.transpose(1, 0)  # w x b x h

        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out

        input_tensor = input_tensor.transpose(1, 0)

        atten_weight = torch.nn.functional.softmax(input_tensor, dim=1)
        atten_weight = atten_weight.expand(x.shape)

        weighted_sum = x * atten_weight
        
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)
    


class LatencyModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(LatencyModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    
class CpuModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(CpuModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    

class MemoryModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(MemoryModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
           
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]

class NetworkOutModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(NetworkOutModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    

class NetworkInModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(NetworkInModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
       
        return hidden_states[:, -1, :]  # [bz, out_dim]

class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()
        '''
        Params:
            in_dim: the feature dim of each node
        '''
        layers = []

        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i-1] if i > 0 else in_dim 
            dropout = kwargs["attn_drop"] if "attn_drop" in kwargs else 0
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head, 
                                        attn_drop=dropout, negative_slope=activation, allow_zero_in_degree=True)) 
            self.maxpool = nn.MaxPool1d(attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1)) 

    
    def forward(self, graph, x):
        '''
        Input:
            x -- tensor float [batch_size*node_num, feature_in_dim] N = {s1, s2, s3, e1, e2, e3}
        '''
        out = None
        for layer in self.net:
            if out is None: out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out) #[bz*node, out_dim] --> [bz, out_dim]


class MultiSourceEncoder(nn.Module):
    def __init__(self, node_num, device, log_dim=64, fuse_dim=64, alpha=0.1, **kwargs):
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = alpha
        self.device = device
        self.low_level_dim = kwargs['input_dims']

        self.latency_model = LatencyModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs)
        latency_dim = self.latency_model.out_dim

        self.cpu_model = CpuModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        cpu_dim = self.cpu_model.out_dim

        self.memory_model = MemoryModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        memory_dim = self.memory_model.out_dim

        self.networkout_model = NetworkOutModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        networkout_dim = self.networkout_model.out_dim

        self.networkin_model = NetworkInModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        networkin_dim = self.networkin_model.out_dim

        fuse_in = latency_dim + cpu_dim + memory_dim + networkout_dim + networkin_dim

        if not fuse_dim % 2 == 0: fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)

        self.activate = nn.GLU()
        self.feat_in_dim = int(fuse_dim // 2)

        self.status_model = GraphModel(in_dim=self.feat_in_dim, device=device, **kwargs)
        self.feat_out_dim = self.status_model.out_dim
    
    def forward(self, graph, latency, cpu, memory, networkout, networkin):
        latency_embedding = self.latency_model(latency) #[bz*node_num, T, trace_dim]
        latency_embedding = latency_embedding.reshape(-1, latency_embedding.size(2))

        cpu_embedding = self.cpu_model(cpu) #[bz*node_num, T, trace_dim]
        cpu_embedding = cpu_embedding.reshape(-1, cpu_embedding.size(2))

        memory_embedding = self.networkout_model(memory) #[bz*node_num, T, trace_dim]
        memory_embedding = memory_embedding.reshape(-1, memory_embedding.size(2))

        networkout_embedding = self.networkout_model(networkout) #[bz*node_num, T, trace_dim]
        networkout_embedding = networkout_embedding.reshape(-1, networkout_embedding.size(2))

        networkin_embedding = self.networkin_model(networkin) #[bz*node_num, T, trace_dim]
        networkin_embedding = networkin_embedding.reshape(-1, networkin_embedding.size(2))


        feature = self.activate(self.fuse(torch.cat((latency_embedding, cpu_embedding, memory_embedding, 
                                                     networkout_embedding, networkin_embedding), dim=-1))) #[bz*node_num, node_dim]

        embeddings = self.status_model(graph, feature) #[bz, graph_dim]
        
        return embeddings


class MainModel(nn.Module):
    def __init__(self, node_num, device, alpha=0.1, **kwargs):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha

        self.encoder = MultiSourceEncoder(self.node_num, device, alpha=alpha, **kwargs)

        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.detector_criterion = nn.CrossEntropyLoss()

        self.localizer1 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer1_criterion = nn.CrossEntropyLoss()

        self.localizer2 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer2_criterion = nn.CrossEntropyLoss()

        self.localizer3 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer3_criterion = nn.CrossEntropyLoss()

        self.localizer4 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer4_criterion = nn.CrossEntropyLoss()

        self.localizer5 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer5_criterion = nn.CrossEntropyLoss()


    def forward(self, graph, latency, cpu, memory, networkout, networkin, labels):  
        batch_size = graph.batch_size
        embeddings = self.encoder(graph, latency, cpu, memory, networkout, networkin) #[bz, feat_out_dim]

        y_window_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local1_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local2_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local3_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local4_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local5_anomaly = torch.zeros(batch_size).long().to(self.device)

        for i in range(batch_size):
            y_window_anomaly[i] = int(labels[i, 0])
            y_local1_anomaly[i] = int(labels[i, 1])
            y_local2_anomaly[i] = int(labels[i, 2])
            y_local3_anomaly[i] = int(labels[i, 3])
            y_local4_anomaly[i] = int(labels[i, 4])
            y_local5_anomaly[i] = int(labels[i, 5])


        detect_logits = self.detector(embeddings)
        detect_loss = self.detector_criterion(detect_logits, y_window_anomaly) 

        locate1_logits = self.localizer1(embeddings)
        locate1_loss = self.localizer1_criterion(locate1_logits, y_local1_anomaly)

        locate2_logits = self.localizer2(embeddings)
        locate2_loss = self.localizer2_criterion(locate2_logits, y_local2_anomaly)

        locate3_logits = self.localizer3(embeddings)
        locate3_loss = self.localizer3_criterion(locate3_logits, y_local1_anomaly)

        locate4_logits = self.localizer4(embeddings)
        locate4_loss = self.localizer4_criterion(locate4_logits, y_local2_anomaly)

        locate5_logits = self.localizer5(embeddings)
        locate5_loss = self.localizer2_criterion(locate5_logits, y_local2_anomaly)


        loss = self.alpha * detect_loss + (1-self.alpha)/5 * locate1_loss + (1-self.alpha)/5 * locate2_loss + (1-self.alpha)/5 * locate3_loss + (1-self.alpha)/5 * locate4_loss + (1-self.alpha)/5 * locate5_loss

        graph_logits, \
            l1_logits, \
                l2_logits, l3_logits, l4_logits, l5_logits = self.inference(batch_size, 
                                                                            detect_logits, 
                                                                            locate1_logits, 
                                                                            locate2_logits,
                                                                            locate3_logits,
                                                                            locate4_logits,
                                                                            locate5_logits)

        return {'loss': loss, 
                'graph_logits': graph_logits, 
                'l1_logits': l1_logits, 
                'l2_logits': l2_logits,
                'l3_logits': l3_logits,
                'l4_logits': l4_logits,
                'l5_logits': l5_logits}

    def inference(self, batch_size, detect_logits=None, locate1_logits=None, locate2_logits=None, locate3_logits=None, locate4_logits=None, locate5_logits=None):
        
        # for i in range(batch_size):
        detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        predictions = torch.tensor(detect_pred)


        locate1_logits = locate1_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l1_predictions = torch.tensor(locate1_logits)


        locate2_logits = locate2_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l2_predictions = torch.tensor(locate2_logits)

        locate3_logits = locate3_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l3_predictions = torch.tensor(locate3_logits)

        locate4_logits = locate4_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l4_predictions = torch.tensor(locate4_logits)

        locate5_logits = locate5_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l5_predictions = torch.tensor(locate5_logits)

        return predictions, l1_predictions, l2_predictions, l3_predictions, l4_predictions, l5_predictions