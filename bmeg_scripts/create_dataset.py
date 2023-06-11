import os
from queue import Queue
import sys
import json
import copy
from collections import defaultdict, deque
import pandas as pd
import pathlib
import csv

def create_folder_p(folder):
    """
    mkdir -p (create parent directories if necessary. Don't throw error if directory already exists)
    """
    pathlib.Path(f"{folder}").mkdir(parents=True, exist_ok=True)

listOfApps = ["SN", "HR", "MM", "TT"]

servNameLookup = {
    'compose': 'nginx-web-server',
    'home':  'nginx-web-server',
    'user': 'nginx-web-server',


    'reviewCompose': 'compose-review-service',
    'reviewRead': 'movie-review-service&operation=ReadMovieReviews',
    'cast': 'cast-info-service&operation=ReadCastInfo',
    'plot': 'plot-service',
}

operation_name_lookup = {
    'compose': '/wrk2-api/post/compose',
    'home': '/wrk2-api/home-timeline/read',
    'user': '/wrk2-api/user-timeline/read'
}

microservice_latency_dict = {}

class TreeNode:
    def __init__(self, spanID, service_name, operation_name, duration, start_time, children, parent = None ):
        self.span_id = spanID
        self.parent = parent
        self.duration = duration
        self.start_time = start_time
        self.service_name = service_name
        self.operation_name = operation_name
        self.children = children


def buildTree( trace, ignore_warnings = True):
    spans = trace['spans']
    processes = trace['processes']

    root = None
    children_of = defaultdict(list)
    for span in spans:
        span_id = span["spanID"]
        if span_id == span["traceID"]: #this is the root node
            root = TreeNode(span_id, processes[span["processID"]]["serviceName"], span["operationName"] , span['duration'],span['startTime'], list())
        else:
            children_of[span["references"][0]["spanID"]].append(span)
    total = 0

    queue = deque([root])
    while queue:
        current = queue.popleft()
        for span in children_of[current.span_id]:
            # each span has only one parent so need to maintain visited set
            if not ignore_warnings and span["warnings"] is not None:
                raise
            node = TreeNode(span["spanID"], processes[span["processID"]]["serviceName"], span["operationName"] , span['duration'],span['startTime'], list())
            queue.append(node)
            current.children.append(node)
        # sort based on process and service name?
        current.children.sort(key = lambda node: node.service_name + "_" + node.operation_name) # sort the children in ascending order of their end time. 
    return root

def levelOrderTraverseTree(root):
    que = Queue()
    que.put(root)
    temp_dict = set()
    repeats = []
    temp_list = []
    level = 0
    while not que.empty():
        print("LEVEL ", level)
        sz = que.qsize()
        for i in range(sz):
            temp = que.get()
            print(temp.service_name, temp.operation_name)
            if (temp.service_name, temp.operation_name) in temp_dict:
                repeats.append((temp.service_name, temp.operation_name))
            else:
                temp_dict.add((temp.service_name, temp.operation_name))
            temp_list.append((temp.service_name, temp.operation_name))
            print("children length ", len(temp.children))
            for child in temp.children:
                que.put(child)
        level += 1
    print(len(temp_dict))
    print(len(temp_list))
    print(repeats)

def is_trace_okay(trace):
    global trace_warnings
    global span_warnings
    for span in trace['spans']:
        if span['warnings'] is not None:
            for warning in span['warnings']:
                if not warning.startswith('invalid parent span ID'):
                    print("New warning message found!! %s" % warning)
            return True
    return True

def getNode(span, trace, spans_dict):
    parent_operation_name = ""
    if len(span['references']) != 0:
        if span['references'][0]['spanID'] != span['spanID']:
            parent_operation_name = spans_dict[f"{trace['traceID']}_{span['references'][0]['spanID']}"]

    if parent_operation_name != "":
        node = trace['processes'][span['processID']]['serviceName'] + "_" + parent_operation_name + "_" + span['operationName']
    else :
        node = trace['processes'][span['processID']]['serviceName'] + "_" + span['operationName']

    return node

def addLatencyOfService(trace, span):
    serviceName = trace['processes'][span['processID']]['serviceName']
    if serviceName not in microservice_latency_dict.keys():
        microservice_latency_dict[serviceName] = []
    else:
        microservice_latency_dict[serviceName].append(span['duration'])


def get_valid_call_graphs(app, request_type):
    valid_trace_folder = f"/home/ubuntu/firm_compass/configs/{app}"
    call_graphs = {}
    counter = 1
    for trace_file in os.listdir(valid_trace_folder):
        if request_type in trace_file:
            with open(os.path.join(valid_trace_folder, trace_file)) as f:
                trace = json.load(f)['data'][0]
                call_graphs[counter] = buildTree(trace)
                counter += 1
    
    return call_graphs

def get_matching_graph(current_graph, valid_graphs):
    def traverse_graphs(current_root, valid_root):
        q1, q2 = deque([current_root]), deque([valid_root])
        
        while q1 and q2:
            current = q1.popleft()
            valid = q2.popleft()
            if current.service_name != valid.service_name or current.operation_name != valid.operation_name or len(current.children) != len(valid.children):
                return False
            for child in current.children:
                q1.append(child)
            for child in valid.children:
                q2.append(child)
        return (not q1) and (not q2) # queues of matching graphs will be empty
    
    for key in valid_graphs:
        if traverse_graphs(current_graph, valid_graphs[key]):
            return key
    
    return -1

def get_paths(root):
    """
    Get paths using BFS as all the other traversals are also in BFS.
    """
    paths = []
    queue = deque([[root, []]])
    id = 0
    while queue:
        cur_node, cur_path = queue.popleft()
        cur_path.append(str(id))
        id += 1
        if not cur_node.children:
            paths.append("->".join(cur_path))
        for child in cur_node.children:
            queue.append([child, cur_path[:]])
    return paths

def append_trace_data(trace , service_time_data, rpc_start_data):
    # node_ids not needed as the order should be the same
    root = buildTree(trace)
    queue = deque([root])
    id = 0
    while queue:
        cur = queue.popleft()
        service_time_data[id].append(cur.duration)
        rpc_start_data[id].append(cur.start_time)
        id += 1
        for child in cur.children:
            queue.append(child)

def write_csv(file, dictionary):
    """
    Dictionary of column name and its values
    """
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))

def write_content(file, data):
    with open(file, "w") as f:
        f.write(data)

def getReqTypeStats(output_folder, reqType, app, readFromFolder, interference_percentage):
    span_length_dict = {"SN": { "compose": 30, "home": 7, "user": 6, }}
    traceFile = str(readFromFolder)+"/"+str(reqType)+"_traces.json"
    os.system("mkdir -p %s" % output_folder)
    with open(traceFile, "r") as storeDataFile:
        traceData = json.load(storeDataFile)['data']
    print("Length of trace data ", len(traceData))

    valid_call_graphs = get_valid_call_graphs("SN", reqType )

    traces_groups = defaultdict(list)

    for trace in traceData:
        current_graph = buildTree(trace)
        key = get_matching_graph(current_graph, valid_call_graphs)
        if key != -1:
            traces_groups[key].append(trace)

    print(len(traces_groups[1]))

    for key in valid_call_graphs:
        graph_paths = get_paths(valid_call_graphs[key])
        print(graph_paths)
        write_content(os.path.join(output_folder, f"graph_paths_{key}"), "\n".join(graph_paths))
        service_time_data = defaultdict(list)
        rpc_start_time_data = defaultdict(list)

        for trace in traces_groups[key]:
            append_trace_data(trace, service_time_data, rpc_start_time_data)
        #levelOrderTraverseTree(valid_call_graphs[key])
        write_csv(os.path.join(output_folder, f"service_time_{key}.csv"), service_time_data)
        write_csv(os.path.join(output_folder, f"rps_start_time_{key}.csv"), rpc_start_time_data)


def main(args):
    applnName = args[1]
    readFromFolder = args[2]
    interference_percentage = args[2]
    opFolder = os.path.join(readFromFolder, "processed_traces")
    create_folder_p(opFolder)
    if(not(applnName in listOfApps)):
        print("\t Appln: %s is not in recognized list: %s " %
              (applnName, listOfApps))
        sys.exit()

    if(applnName == "SN"):
        getReqTypeStats(opFolder, 'compose', applnName, readFromFolder, interference_percentage)
        #getReqTypeStats(opFolder, 'home', applnName, readFromFolder, interference_percentage)
        #getReqTypeStats(opFolder, 'user', applnName, readFromFolder, interference_percentage)


if __name__ == "__main__":
    main(sys.argv)
    # for k, v in microservice_latency_dict.items():
    #     series = pd.Series(v)
    #     print("Service : " , k)
    #     print(series.describe())
