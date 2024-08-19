import os
import sys
import numpy as np
import pandas as pd
from utils import read_mapping_From_VM_to_microservice, resource_files, resource_to_column_mapping, metrics
import os
from create_paths import create_paths
import json
from SEER_create_dataset_trace import createMultiModalDataset
import random
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gtype = "g1"

source_pkl = "./util_data/source.pkl"
target_pkl = "./util_data/target.pkl"

with open(source_pkl, 'rb') as s:
    source = pickle.load(s)
s.close()

with open(target_pkl, 'rb') as t:
    target = pickle.load(t)
t.close()

window_size = [10, 15, 20, 25, 30, 35, 40, 45, 50]
train_prob = 0.85
microservices = 30

random.seed(12345)

## generate multi modals files and labels
# _,_ = createMultiModalDataset()

directory_path = "./multi-modal-data-separate"
# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

features = ["latency", "container_cpu_usage_seconds_total", "container_memory_usage_bytes", 
            "container_network_transmit_bytes_total", "container_network_receive_bytes_total"]
col_label = "label_RPC"

## compose machine to microservice mapping
ms238 = [13, 14, 21, 26, 27, 28, 7, 8, 11, 12, 18, 19, 4, 5]
ms47 = [22, 0, 1, 2]
ms28 = [13, 14, 21, 26, 27, 28, 4, 5]
ms247 = [13, 14, 21, 26, 27, 22, 0, 1, 2]
ms378 = [7, 8, 11, 12, 18, 19, 0, 1, 2, 4, 5]

## home machine to microservice mapping
# ms238 = [3, 4]
# ms47 = [0, 1, 2, 6]
# ms28 = [4]
# ms247 = [0, 1, 2, 4, 6]
# ms378 = [0, 1, 2, 4, 6]
## user machine to microservice mapping
# ms47 = [0, 1, 2, 8]

# print(source)
# print(target)
# sys.exit(0)

if __name__ == "__main__":

    
    # for i in range(30):
    #     microservices_order.append(str(i))
    
    # microservices_order = sorted(microservices_order)
    # print(microservices_order)


    for w in window_size:
        microservices_order = []
        full_data = []
        train_data = []
        test_data = []

        window_time_count = []
        metadata = {}

        count_invalid_windows = 0
        count_valid_windows = 0
        count_train_windows = 0
        count_test_windows = 0
        all_windows = 0

        count_anomalies = 0
        count_nonanomalies = 0
        count_238 = 0
        count_47 = 0
        count_247 = 0
        count_28 = 0
        count_378 = 0


        print("window size: ", w)
        for csv_file in csv_files:
            print(csv_file)
            
            csv_string = os.path.splitext(csv_file)[0]
            df = pd.read_csv(os.path.join(directory_path, csv_file))

            df['0_start'] = pd.to_datetime(df['0_start'].astype(int), unit='us')
            df = df.sort_values(by='0_start')
            total_rows = df.shape[0]
            window_ids = np.arange(total_rows) // w
            # print(window_ids)
            df['window_id'] = csv_string + '_' + pd.Series(window_ids).astype(str)
            # start_time = df['0_start'].min()
            # df['time_diff'] = (df['0_start'] - start_time).dt.total_seconds()
            # df['window'] = df['time_diff'] // window_size
            # df['window'] = df['window'].astype(int)
            # df['window_id'] = csv_string + '_' + df['window'].astype(str)
            # df = df.drop(columns = ['window'])
            # df = df.drop(columns = ['time_diff'])

            # all_csv.append(df)

            grouped = df.groupby('window_id')

            toggle_full_data = 1

            for window_id, group in grouped:
                toggle_full_data = 1
                if group.shape[0] != w:
                    toggle_full_data = 0
                all_windows += 1
                # toggle_add_window = 1
                # print(group, type(group))
                # sys.exit(0)
                tt_prob = random.random()
                count_features = 0
                start_time = group['0_start'].min()
                end_time = group['0_start'].max()
                time_diff = (end_time - start_time).total_seconds()
                

                time_cols = group.filter(like='start', axis=1).columns
                group = group.drop(columns=time_cols)
                group = group.sort_index(axis=1)


                data = {}
                data['window_id'] = window_id
                for f in features:
                    count_features += 1
                    feature_df = group.filter(like=f) 
                    feature_col = sorted(feature_df.columns)
                    feature_df = feature_df[feature_col]

                    if count_features == 1:
                        microservices_order = [int(item.split('_')[0]) for item in feature_col]
                    feature_df.columns = microservices_order
                    feature_ncol = sorted(feature_df.columns)
                    feature_df = feature_df[feature_ncol]
                    # print(feature_df, feature_ncol)
                    # sys.exit(0)
                    # print(feature_df.shape, feature_col)
                    data[f] = feature_df
                
                data['microservices_order'] = microservices_order
                label_df = group.filter(like=col_label)
                label_col = sorted(label_df.columns)
                label_df = label_df[label_col]

                # print(label_df)
                toggle_window = 1 if all(label_df[col].nunique() == 1 for col in label_df.columns) else 0
                label_ms = [label_df[col].unique().tolist()[0] for col in label_df.columns]
                data['label_microservices'] = label_ms

                col238 = [str(i) + '_' + col_label for i in ms238]
                col47 = [str(i) + '_' + col_label for i in ms47]
                col247 = [str(i) + '_' + col_label for i in ms247]
                col28 = [str(i) + '_' + col_label for i in ms28]
                col378 = [str(i) + '_' + col_label for i in ms378]

                label_238 = 1 if all(label_df[col].unique().tolist()[0] == 1 for col in col238) else 0
                label_47 = 1 if all(label_df[col].unique().tolist()[0] == 1 for col in col47) else 0
                label_247 = 1 if all(label_df[col].unique().tolist()[0] == 1 for col in col247) else 0
                label_28 = 1 if all(label_df[col].unique().tolist()[0] == 1 for col in col28) else 0
                label_378 = 1 if all(label_df[col].unique().tolist()[0] == 1 for col in col378) else 0

                lcol_238 = 0
                lcol_47 = 0
                lcol_247 = 0
                lcol_28 = 0
                lcol_378 = 0

                if label_238 == 1 and label_28 == 1:
                    lcol_238 = 1
                    count_238 += 1
                if label_238 == 0 and label_28 == 1:
                    lcol_28 = 1   
                    count_28 += 1 
                if label_247 == 1 and label_47 == 1:
                    lcol_247 = 1 
                    count_247 += 1   
                if label_247 == 0 and label_47 == 1:
                    lcol_47 = 1   
                    count_47 += 1 
                if label_378 == 1:
                    lcol_378 = 1
                    count_378 += 1

                window_label = 0
                if (label_df == 1).any().any():
                    window_label = 1
                    count_anomalies += 1
                else:
                    count_nonanomalies += 1
                    # print(window_label, [lcol_238, lcol_47, lcol_247, lcol_28])

                data['label_window'] = window_label
                data['label_window_238'] = lcol_238
                data['label_window_47'] = lcol_47
                data['label_window_247'] = lcol_247
                data['label_window_28'] = lcol_28
                data['label_window_378'] = lcol_378
                data['label_ms'] = label_ms
                data['total_rows'] = group.shape[0]
                data['source'] = source
                data['target'] = target

                if toggle_window == 1 and toggle_full_data == 1:
                    full_data.append(data)
                    window_time_count.append(time_diff)
                    count_valid_windows += 1
                    if tt_prob > train_prob:
                        test_data.append(data)
                        count_test_windows += 1
                    else:
                        train_data.append(data)
                        count_train_windows += 1
                else:
                    count_invalid_windows += 1
                
                # window_row_count.append(group.shape[0])

                # if 1 in [label_238, label_47, label_247, label_28]:
                #     print(label_238, label_47, label_247, label_28)

                # if toggle_window == 0:
                #     print("window_id: ", window_id, " is invalid")
                #     count_invalid_windows += 1
                #     toggle_add_window = 0

                # print(toggle_window)
                # sys.exit(0)


                # print(microservices_order, label_col)

                # sys.exit(0)
                # if tt_prob > train_prob:
                    
                # else:
        print("total windows: ", all_windows)
        print("total valid windows: ", count_valid_windows)
        print("total invalid windows: ", count_invalid_windows)
        print("total train windows: ", count_train_windows)
        print("total test windows: ", count_test_windows)

        metadata['window_interval'] = w
        metadata['max_window_time'] = max(window_time_count)
        metadata['min_window_time'] = min(window_time_count)
        metadata['avg_window_time'] = sum(window_time_count) / len(window_time_count)
        
        metadata['1% window_time'] = np.percentile(window_time_count, 1)
        metadata['25% window_time'] = np.percentile(window_time_count, 25)
        metadata['median_window_time'] = np.median(window_time_count)
        metadata['75% window_time'] = np.percentile(window_time_count, 75)
        metadata['99% window_time'] = np.percentile(window_time_count, 99)


        metadata['total_windows'] = all_windows
        metadata['total_valid_windows'] = count_valid_windows
        metadata['total_invalid_windows'] = count_invalid_windows
        metadata['total_train_windows'] = count_train_windows
        metadata['total_test_windows'] = count_test_windows
        metadata['total anomalies'] = count_anomalies
        metadata['total non-anomalies'] = count_nonanomalies
        metadata['total 238'] = count_238
        metadata['total 47'] = count_47
        metadata['total 247'] = count_247
        metadata['total 28'] = count_28
        metadata['total 378'] = count_378
        metadata['source'] = source
        metadata['target'] = target
        

        print("metadata : ", metadata)

        save_data = ''.join(["./util_data/", "window_features_" + gtype + "_" + str(w)])
        if not os.path.exists(save_data):
            os.makedirs(save_data)

        metadata_json_path = os.path.join(save_data, "metadata.json")

        with open(metadata_json_path, 'w') as metadata_json:
            json.dump(metadata, metadata_json, indent=4)
        metadata_json.close()

        full_pkl_path = os.path.join(save_data, "full_data.pkl")
        with open(full_pkl_path, 'wb') as full_pkl:
            pickle.dump(full_data, full_pkl)

        train_pkl_path = os.path.join(save_data, "train_data.pkl")
        with open(train_pkl_path, 'wb') as train_pkl:
            pickle.dump(train_data, train_pkl)

        test_pkl_path = os.path.join(save_data, "test_data.pkl")
        with open(test_pkl_path, 'wb') as test_pkl:
            pickle.dump(test_data, test_pkl)  

        train_sample = random.sample(train_data, 100)  
        test_sample = random.sample(test_data, 10)  

        train_sample_path = os.path.join(save_data, "train_sample.pkl")
        with open(train_sample_path, 'wb') as train_s:
            pickle.dump(train_sample, train_s)
        
        test_sample_path = os.path.join(save_data, "test_sample.pkl")
        with open(test_sample_path, 'wb') as test_s:
            pickle.dump(test_sample, test_s)

    # train_file = "util_data/window_features_json_" + str(window_size) + "/train_data.json"
    # data = pd.read_json("util_data/window_features_json_0.5/train_data.json")
    # for window_id, window_data in data.items():
    #     for microservice, features in window_data.items():
    #         for feature in features.keys():
                
    #         print(window_id, microservice, features.keys())
    #         sys.exit(0)