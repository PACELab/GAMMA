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

window_size = 0.5
train_prob = 0.85
microservices = 30

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(12345)

## generate multi modals files and labels
_,_ = createMultiModalDataset()

directory_path = "./multi-modal-data-separate"
# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Read and append each CSV
# all_dataframes = []

save_folder = ''.join(["./util_data/", "window_features_" + str(window_size)])
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_json = ''.join(["./util_data/", "window_features_json_" + str(window_size)])
if not os.path.exists(save_json):
    os.makedirs(save_json)

full_json_path = os.path.join(save_json, "full_data.json")
train_json_path = os.path.join(save_json, "train_data.json")
test_json_path = os.path.join(save_json, "test_data.json")

full_label_path = os.path.join(save_json, "full_label.json")
train_label_path = os.path.join(save_json, "train_label.json")
test_label_path = os.path.join(save_json, "test_label.json")


features = ["latency", "container_cpu_usage_seconds_total", "container_memory_usage_bytes", 
            "container_network_transmit_bytes_total", "container_network_receive_bytes_total"]
col_label = "label_RPC"

full_data = {}
train_data = {}
test_data = {}

full_label = {}
train_label = {}
test_label = {}

full_window_label = {}
train_window_label = {}
test_window_label = {}

all_csv = []
train_csv = []
test_csv = []

all_count = 0
train_count = 0
test_count = 0

for csv_file in csv_files:
    print(csv_file)
    csv_string = os.path.splitext(csv_file)[0]
    df = pd.read_csv(os.path.join(directory_path, csv_file))

    df['0_start'] = pd.to_datetime(df['0_start'].astype(int), unit='us')
    df = df.sort_values(by='0_start')
    start_time = df['0_start'].min()
    df['time_diff'] = (df['0_start'] - start_time).dt.total_seconds()
    df['window'] = df['time_diff'] // window_size
    df['window'] = df['window'].astype(int)
    df['window_id'] = csv_string + '_' + df['window'].astype(str)
    df = df.drop(columns = ['window'])
    df = df.drop(columns = ['time_diff'])
    time_cols = df.filter(like='start', axis=1).columns
    df = df.drop(columns=time_cols)
    df = df.sort_index(axis=1)

    all_csv.append(df)

    grouped = df.groupby('window_id')

    count_invalid_windows = 0
    count_valid_windows = 0

    for window_id, group in grouped:
        all_count += 1
        # print(group, type(group))
        # sys.exit(0)
        full_data[window_id] = {}
        full_label[window_id] = {}
        full_window_label[window_id] = {}

        tt_prob = random.random()
        if tt_prob > train_prob:
            test_csv.append(group)
            test_count += 1
            test_data[window_id] = {}
            test_label[window_id] = {}
            test_window_label[window_id] = {}
        else:
            train_csv.append(group)
            train_count += 1
            train_data[window_id] = {}
            train_label[window_id] = {}
            train_window_label[window_id] = {}

        for m in range(microservices):
            full_data[window_id][m] = {}
            full_label[window_id][m] = {}
            full_window_label[window_id][m] = {}
            if tt_prob > train_prob:
                test_data[window_id][m] = {}
                test_label[window_id][m] = {}
                test_window_label[window_id][m] = {}
            else:
                train_data[window_id][m] = {}
                train_label[window_id][m] = {}
                train_window_label[window_id][m] = {}

            clab = str(m) + "_" + col_label
            all_labels = np.unique(group.loc[:, clab].values)

            if len(all_labels) == 1:
                for f in features:
                    col = str(m) + "_" + f
                    # print(col, group.loc[:, col].values)
                    full_data[window_id][m][f] = group.loc[:, col].values.tolist()
                    full_window_label[window_id][m][f] = all_labels[0]
                    if tt_prob > train_prob:
                        test_data[window_id][m][f] = group.loc[:, col].values.tolist()
                        test_window_label[window_id][m][f] = all_labels[0]
                    else:
                        train_data[window_id][m][f] = group.loc[:, col].values.tolist()
                        train_window_label[window_id][m][f] = all_labels[0]
                
                full_label[window_id][m][f] = group.loc[:, clab].values.tolist()
                if tt_prob > train_prob:
                    test_label[window_id][m][f] = group.loc[:, clab].values.tolist()
                else:
                    train_label[window_id][m][f] = group.loc[:, clab].values.tolist()
                count_valid_windows += 1
            else:
                count_invalid_windows += 1



print("All windows : ", all_count)
print("Valid Train windows : ", train_count)
print("Valid Test windows : ", test_count)

final_dataframe = pd.concat(all_csv, ignore_index=True)
df_save = os.path.join(save_json, "full_data.csv")
final_dataframe.to_csv(df_save, index = False)

train_dataframe = pd.concat(train_csv, ignore_index=True)
df_save = os.path.join(save_json, "train_data.csv")
train_dataframe.to_csv(df_save, index = False)

test_dataframe = pd.concat(test_csv, ignore_index=True)
df_save = os.path.join(save_json, "test_data.csv")
test_dataframe.to_csv(df_save, index = False)


full_json_path = os.path.join(save_json, "full_data.json")
train_json_path = os.path.join(save_json, "train_data.json")
test_json_path = os.path.join(save_json, "test_data.json")


with open(full_json_path, 'w') as full_json:
    json.dump(full_data, full_json, indent=4)
full_json.close()

with open(train_json_path, 'w') as train_json:
    json.dump(train_data, train_json, indent=4)
train_json.close()

with open(test_json_path, 'w') as test_json:
    json.dump(test_data, test_json, indent=4)
test_json.close()


with open(full_label_path, 'w') as full_l:
    json.dump(full_label, full_l, indent=4)
full_l.close()

with open(train_label_path, 'w') as train_l:
    json.dump(train_label, train_l, indent=4)
train_l.close()

with open(test_label_path, 'w') as test_l:
    json.dump(test_label, test_l, indent=4)
test_l.close()

