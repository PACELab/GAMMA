import os
from queue import Queue
import sys
import json
import copy
from collections import defaultdict
import pandas as pd

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
            parent_operation_name = spans_dict[trace['traceID']+span['references'][0]['spanID']]

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


def getReqTypeStats(opFolder, reqType, app, readFromFolder, interference_percentage):
    span_length_dict = {"SN": { "compose": 10, "home": 4, "user": 6, }}
    traceFile = str(readFromFolder)+"/"+str(reqType)+"_traces.txt"
    output_folder = str(opFolder) + "/bottleneck_dataset/" + str(readFromFolder) + "/"
    os.system("mkdir -p %s" % output_folder)
    with open(traceFile, "r") as storeDataFile:
        traceData = json.load(storeDataFile)['data']
    print("Length of trace data ", len(traceData))
    fileNamePrefix = output_folder + str(reqType)

    node_to_id = {}
    id_to_node = {}
    orderingMap = {}
    orderingTrace = {}
    iterator = 0

    """
    Maintaining a spans_dict to keep track of parent trace_span operation_names.
    """
    spans_dict = defaultdict(str)
    for trace in traceData:
        for span in trace['spans']:
            spans_dict[trace['traceID']+span['spanID']] = span['operationName']

    # assign an ID to each node (serviceName_operationName) so that call graphs can be uniquely identified by the ordering of the IDs
    for trace in traceData:
        for span in trace['spans']:
            # service names (microservice names) are replaced by IDs of the form p1, p2, p3 and the mapping is present in trace['processes']
            node = getNode(span, trace, spans_dict)
            if node not in node_to_id:
                node_to_id[node] = iterator
                id_to_node[iterator] = node
                iterator += 1

    node_to_id['start_time'] = iterator
    id_to_node[iterator] = 'start_time'
    iterator +=1

    node_to_id['interference_percentage'] = iterator
    id_to_node[iterator] = 'interference_percentage'
    
    # filter trace data by removing the ones with warnings:
    filtered_trace_data = []
    for trace in traceData:
        if is_trace_okay(trace):
            filtered_trace_data.append(trace)

    print(len(filtered_trace_data))
    count = 0
    for trace in filtered_trace_data:
        ordering = set()
        # traversing over all the spans (so no backtracking required)
        for span in trace['spans']:
            node = getNode(span, trace, spans_dict)
            ordering.add(node_to_id[node])
        ordering.add(node_to_id['start_time'])
        ordering.add(node_to_id['interference_percentage'])

        # To make it hashable
        ordering = frozenset(sorted(ordering))
        # some of the traces might having missing spans.
        if ordering not in orderingMap and len(ordering) >=  span_length_dict[app][reqType]:
            count += 1
            orderingMap[ordering] = count
            orderingTrace[ordering] = trace

    print(orderingMap)
    
    for ordering, idx in orderingMap.items():
        #TODO: Create a file similar to FIRM dataset that shows the graph structure.
        # 1->2->4
        # 1->3->5
        # 1->2->6
        print("Filename ", fileNamePrefix + '_' + str(idx))
        opFile = open(fileNamePrefix + '_' + str(idx), "w+")
        traceFile = open(fileNamePrefix + '_trace_' + str(idx) , "w+")
        trace = orderingTrace[ordering]
        ordering = list(ordering)
        print("Ordering list before printing columns", ordering)
        for id in ordering:
            #print("id_to_node[id]", id_to_node[id])
            opFile.write(str(id_to_node[id])+",")
        json.dump(trace, traceFile)
        opFile.write("\n")
        opFile.close()

    for trace in filtered_trace_data:
        ordering_latency_dict = {}
        for span in trace['spans']:
            node = getNode(span, trace, spans_dict)
            # if a microservice_method is called multiple times, add their latencies.
            addLatencyOfService(trace, span)
            if node_to_id[node] not in ordering_latency_dict:
                ordering_latency_dict[node_to_id[node]] = span['duration']
            else:
                print(node_to_id[node])
                d = ordering_latency_dict.get(node_to_id[node])

        ordering_latency_dict[node_to_id['start_time']] = span['startTime']
        ordering_latency_dict[node_to_id['interference_percentage']] = interference_percentage

        sorted_keys = sorted(list(ordering_latency_dict.keys()))
        ordering = frozenset(sorted_keys)
        if ordering in orderingMap:
            idx = orderingMap[ordering]
            opFile = open(fileNamePrefix + '_' + str(idx), "a")
            for item in sorted_keys:
                opFile.write(str(ordering_latency_dict[item])+",")
            opFile.write("\n")
            opFile.close()

def main(args):
    opFolder = args[1]
    applnName = args[2]
    readFromFolder = args[3]
    interference_percentage = args[4]

    if(not(applnName in listOfApps)):
        print("\t Appln: %s is not in recognized list: %s " %
              (applnName, listOfApps))
        sys.exit()

    if(applnName == "SN"):
        getReqTypeStats(opFolder, 'compose', applnName, readFromFolder, interference_percentage)
        getReqTypeStats(opFolder, 'home', applnName, readFromFolder, interference_percentage)
        getReqTypeStats(opFolder, 'user', applnName, readFromFolder, interference_percentage)


if __name__ == "__main__":
    main(sys.argv)
    for k, v in microservice_latency_dict.items():
        series = pd.Series(v)
        print("Service : " , k)
        print(series.describe())
