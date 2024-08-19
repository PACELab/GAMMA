import os
import time
import copy
import sys
import torch
from torch import nn
import logging
from utils import *
from models import MainModel
from models_cpu import EvaluateCpu
from models_memory import EvaluateMemory
from models_network import EvaluateNetwork
from models_mixed import EvaluateMixed

class BaseModel(nn.Module):
    def __init__(self, node_num, device, lr=1e-3, epochs=20, result_dir='./results', **kwargs):
        super(BaseModel, self).__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.node_num = node_num
        self.model_save_dir = kwargs['model_save_dir']

        # self.model_save_dir = os.path.join(result_dir, hash_id)
        if kwargs['model'] == 'all':
            self.model = MainModel(self.node_num, self.device, alpha=0.2, **kwargs)
        elif kwargs['model'] == 'cpu':
            self.model = EvaluateCpu(self.node_num, self.device, alpha=0.2, **kwargs)
        elif kwargs['memory'] == 'memory':
            self.model = EvaluateMemory(self.node_num, self.device, alpha=0.2, **kwargs)
        elif kwargs['network'] == 'network':
            self.model = EvaluateNetwork(self.node_num, self.device, alpha=0.2, **kwargs)
        elif kwargs['mixed'] == 'mixed':
            self.model = EvaluateMixed(self.node_num, self.device, alpha=0.2, **kwargs)
        else:
            print("Please select a valid model")
            sys.exit(1)
        self.model.to(device)

    def save_model(self, state, model_save_dir = None):
        if model_save_dir is None: 
            file = "./model.pth"
        else:
            file = os.path.join(model_save_dir, "model.pth")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=True)
        except:
            torch.save(state, file, _use_new_zipfile_serialization=False)


    def load_model(self, model, model_save_dir = None):
        if model_save_dir is None: 
            file = "./model.pth"
        else:
            file = os.path.join(model_save_dir, "model.pth")
        
        model.load_state_dict(torch.load(file))
        return model
    

    def evaluate(self, test_loader):
        self.load_model(self.model, model_save_dir = self.model_save_dir)
        batch_cnt, epoch_loss = 0, 0.0
        self.model.eval()

        with torch.no_grad():
            
            graph_true = []
            l1_true = []
            l2_true = []
            l3_true = []
            l4_true = []
            l5_true = []

            graph_pred = []
            l1_pred = []
            l2_pred = []
            l3_pred = []
            l4_pred = []
            l5_pred = []

            for graph, l, c, m, nout, nin, label in test_loader:
                
                y_w_anomaly = []
                y_l1_anomaly = []
                y_l2_anomaly = []
                y_l3_anomaly = []
                y_l4_anomaly = []
                y_l5_anomaly = []

                for i, _ in enumerate(label):

                    y_w_anomaly.append(int(label[i, 0]))
                    y_l1_anomaly.append(int(label[i, 1]))
                    y_l2_anomaly.append(int(label[i, 2]))
                    y_l3_anomaly.append(int(label[i, 3]))
                    y_l4_anomaly.append(int(label[i, 4]))
                    y_l5_anomaly.append(int(label[i, 5]))
                    
                graph_true.extend(y_w_anomaly)
                l1_true.extend(y_l1_anomaly)
                l2_true.extend(y_l2_anomaly)
                l3_true.extend(y_l3_anomaly)
                l4_true.extend(y_l4_anomaly)
                l5_true.extend(y_l5_anomaly)

                pred = self.model.forward(graph.to(self.device), l.to(self.device), c.to(self.device), m.to(self.device), nout.to(self.device), nin.to(self.device), label.to(self.device))
                loss = pred['loss']
                batch_cnt += 1
                epoch_loss += loss.item()

                graph_logits = pred['graph_logits'].tolist()
                l1_logits = pred['l1_logits'].tolist()
                l2_logits = pred['l2_logits'].tolist()
                l3_logits = pred['l3_logits'].tolist()
                l4_logits = pred['l4_logits'].tolist()
                l5_logits = pred['l5_logits'].tolist()


                graph_pred.extend(graph_logits)
                l1_pred.extend(l1_logits)
                l2_pred.extend(l2_logits)
                l3_pred.extend(l3_logits)
                l4_pred.extend(l4_logits)
                l5_pred.extend(l5_logits)

            epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1 = get_metrics(graph_pred, graph_true)
            epoch_l1_accuracy, epoch_l1_precision, epoch_l1_recall, epoch_l1_f1  = get_metrics(l1_pred, l1_true)
            epoch_l2_accuracy, epoch_l2_precision, epoch_l2_recall, epoch_l2_f1 = get_metrics(l2_pred, l2_true)
            epoch_l3_accuracy, epoch_l3_precision, epoch_l3_recall, epoch_l3_f1 = get_metrics(l3_pred, l3_true)
            epoch_l4_accuracy, epoch_l4_precision, epoch_l4_recall, epoch_l4_f1 = get_metrics(l4_pred, l4_true)
            epoch_l5_accuracy, epoch_l5_precision, epoch_l5_recall, epoch_l5_f1 = get_metrics(l5_pred, l5_true)

            epoch_loss = epoch_loss / batch_cnt
            

            print("testing loss: {:.5f}, testing accuracy: {:.2f}, testing precision: {:.2f}, testing recall: {:.2f}, testing f1: {:.2f} ".format(epoch_loss, epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1))
            print("testing loss: {:.5f}, l1 testing accuracy: {:.2f}, l1 testing precision: {:.2f}, l1 testing recall: {:.2f}, l1 testing f1: {:.2f} ".format(epoch_loss, epoch_l1_accuracy, epoch_l1_precision, epoch_l1_recall, epoch_l1_f1))
            print("testing loss: {:.5f}, l2 testing accuracy: {:.2f}, l2 testing precision: {:.2f}, l2 testing recall: {:.2f}, l2 testing f1: {:.2f} ".format(epoch_loss, epoch_l2_accuracy, epoch_l2_precision, epoch_l2_recall, epoch_l2_f1))
            print("testing loss: {:.5f}, l3 testing accuracy: {:.2f}, l3 testing precision: {:.2f}, l3 testing recall: {:.2f}, l3 testing f1: {:.2f} ".format(epoch_loss, epoch_l3_accuracy, epoch_l3_precision, epoch_l3_recall, epoch_l3_f1))
            print("testing loss: {:.5f}, l4 testing accuracy: {:.2f}, l4 testing precision: {:.2f}, l4 testing recall: {:.2f}, l4 testing f1: {:.2f} ".format(epoch_loss, epoch_l4_accuracy, epoch_l4_precision, epoch_l4_recall, epoch_l4_f1))
            print("testing loss: {:.5f}, l5 testing accuracy: {:.2f}, l5 testing precision: {:.2f}, l5 testing recall: {:.2f}, l5 testing f1: {:.2f} ".format(epoch_loss, epoch_l5_accuracy, epoch_l5_precision, epoch_l5_recall, epoch_l5_f1))
            logging.info("testing loss: {:.5f}, testing accuracy: {:.2f}, testing precision: {:.2f}, testing recall: {:.2f}, testing f1: {:.2f} ".format(epoch_loss, epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1))
            # print("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed))
            print("******************************************************************")

        return [graph_pred, graph_true, l1_pred, l1_true, l2_pred, l2_true, l3_pred, l3_true, l4_pred, l4_true, l5_pred, l5_true]

    def fit(self, train_loader):
    ## initializing the fit function

        best_hr1, coverage, best_state, eval_res = -1, None, None, None # evaluation
        pre_loss, worse_count = float("inf"), 0 # early break

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            
            graph_true = []
            l1_true = []
            l2_true = []
            l3_true = []
            l4_true = []
            l5_true = []

            graph_pred = []
            l1_pred = []
            l2_pred = []
            l3_pred = []
            l4_pred = []
            l5_pred = []
    
            for graph, l, c, m, nout, nin, label in train_loader:

                y_w_anomaly = []
                y_l1_anomaly = []
                y_l2_anomaly = []
                y_l3_anomaly = []
                y_l4_anomaly = []
                y_l5_anomaly = []

                for i, _ in enumerate(label):

                    y_w_anomaly.append(int(label[i, 0]))
                    y_l1_anomaly.append(int(label[i, 1]))
                    y_l2_anomaly.append(int(label[i, 2]))
                    y_l3_anomaly.append(int(label[i, 3]))
                    y_l4_anomaly.append(int(label[i, 4]))
                    y_l5_anomaly.append(int(label[i, 5]))
                
                graph_true.extend(y_w_anomaly)
                l1_true.extend(y_l1_anomaly)
                l2_true.extend(y_l2_anomaly)
                l3_true.extend(y_l3_anomaly)
                l4_true.extend(y_l4_anomaly)
                l5_true.extend(y_l5_anomaly)

                
                optimizer.zero_grad()
                sgd = self.model.forward(graph.to(self.device), l.to(self.device), c.to(self.device), m.to(self.device), nout.to(self.device), nin.to(self.device), label.to(self.device))
                loss = sgd['loss']
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

                graph_logits = sgd['graph_logits'].tolist()
                l1_logits = sgd['l1_logits'].tolist()
                l2_logits = sgd['l2_logits'].tolist()
                l3_logits = sgd['l3_logits'].tolist()
                l4_logits = sgd['l4_logits'].tolist()
                l5_logits = sgd['l5_logits'].tolist()


                graph_pred.extend(graph_logits)
                l1_pred.extend(l1_logits)
                l2_pred.extend(l2_logits)
                l3_pred.extend(l3_logits)
                l4_pred.extend(l4_logits)
                l5_pred.extend(l5_logits)

            epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1 = get_metrics(graph_pred, graph_true)
            epoch_l1_accuracy, epoch_l1_precision, epoch_l1_recall, epoch_l1_f1  = get_metrics(l1_pred, l1_true)
            epoch_l2_accuracy, epoch_l2_precision, epoch_l2_recall, epoch_l2_f1 = get_metrics(l2_pred, l2_true)
            epoch_l3_accuracy, epoch_l3_precision, epoch_l3_recall, epoch_l3_f1 = get_metrics(l3_pred, l3_true)
            epoch_l4_accuracy, epoch_l4_precision, epoch_l4_recall, epoch_l4_f1 = get_metrics(l4_pred, l4_true)
            epoch_l5_accuracy, epoch_l5_precision, epoch_l5_recall, epoch_l5_f1 = get_metrics(l5_pred, l5_true)

            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            

            print("Epoch {}/{}, training loss: {:.5f}, training accuracy: {:.2f}, training precision: {:.2f}, training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1, epoch_time_elapsed))
            print("Epoch {}/{}, training loss: {:.5f}, l1 training accuracy: {:.2f}, l1 training precision: {:.2f}, l1 training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_l1_accuracy, epoch_l1_precision, epoch_l1_recall, epoch_l1_f1, epoch_time_elapsed))
            print("Epoch {}/{}, training loss: {:.5f}, l2 training accuracy: {:.2f}, l2 training precision: {:.2f}, l2 training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_l2_accuracy, epoch_l2_precision, epoch_l2_recall, epoch_l2_f1, epoch_time_elapsed))
            print("Epoch {}/{}, training loss: {:.5f}, l3 training accuracy: {:.2f}, l3 training precision: {:.2f}, l3 training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_l3_accuracy, epoch_l3_precision, epoch_l3_recall, epoch_l3_f1, epoch_time_elapsed))
            print("Epoch {}/{}, training loss: {:.5f}, l4 training accuracy: {:.2f}, l4 training precision: {:.2f}, l4 training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_l4_accuracy, epoch_l4_precision, epoch_l4_recall, epoch_l4_f1, epoch_time_elapsed))
            print("Epoch {}/{}, training loss: {:.5f}, l5 training accuracy: {:.2f}, l5 training precision: {:.2f}, l5 training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_l5_accuracy, epoch_l5_precision, epoch_l5_recall, epoch_l5_f1, epoch_time_elapsed))

            logging.info("Epoch {}/{}, training loss: {:.5f}, training accuracy: {:.2f}, training precision: {:.2f}, training recall: {:.2f}, training f1: {:.2f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_graph_accuracy, epoch_graph_precision, epoch_graph_recall, epoch_graph_f1, epoch_time_elapsed))
            # print("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed))
            print("******************************************************************")

            best_state = copy.deepcopy(self.model.state_dict())
            model_path = self.model_save_dir
            self.save_model(best_state, model_path)

        return [graph_pred, graph_true, l1_pred, l1_true, l2_pred, l2_true, l3_pred, l3_true, l4_pred, l4_true, l5_pred, l5_true]

