# import requests
# import json

# servNameLookup = {
#     'compose': 'nginx-web-server',
#     'home' :  'nginx-web-server',
#     'user' : 'nginx-web-server' ,

#     'reviewCompose' : 'compose-review-service',
#     'reviewRead' : 'movie-review-service&operation=ReadMovieReviews',
#     'cast' : 'cast-info-service&operation=ReadCastInfo',
#     'plot' : 'plot-service',
# }

# operation_name_lookup = {
#     'compose' : '/wrk2-api/post/compose',
#     'home' : '/wrk2-api/home-timeline/read',
#     'user' : '/wrk2-api/user-timeline/read'  
# }
# def get_traces(end_time, start_time, request_type):
#     urlPrefix = "http://localhost:16686/api/traces?limit=10&end="+str(end_time)+"&start="+str(start_time)
#     urlSuffix = "&operation="+operation_name_lookup[request_type] + "&service="+str(servNameLookup[request_type] )
#     url = str(urlPrefix)+str(urlSuffix)
#     print ("\t url: %s "%(url))
#     res = requests.get(url)
#     print(res)
#     print( json.loads(res.content.decode('utf-8')))

# #http://130.245.169.45:16686/search?end=1685134800000000&limit=10&lookback=custom&maxDuration&minDuration&operation=%2Fwrk2-api%2Fpost%2Fcompose&service=nginx-web-server&start=1684530000000000
# get_traces(1685244052263534, 1685300648860000, "compose")

import requests
import pandas as pd
import logging
import json
import time
import os
import sys


def extract_traces(service_name, traces):
    trace_data = []
    process_map = traces["processes"]
    for trace in traces:
        trace_dict = {}
        for span in trace['spans']:
            process_id = span["processID"]
            process = process_map[process_id]
            service = process["serviceName"]
            container = process["tags"][0]["value"]
            operation = span['operationName']
            parent = span['references'][0]['spanID'] if span['references'] else ''
            service_key = f"{service}:{operation}:{parent}"
            if service_key in trace_dict:
                trace_dict[service_key] += 1
            else:
                trace_dict[service_key] = 1
        trace_data.append(trace_dict)
    return trace_data

def generate_csv(trace_data):
    services = set()
    for trace in trace_data:
        services.update(trace.keys())

    df = pd.DataFrame(columns=services)
    for trace in trace_data:
        df = df.append(trace, ignore_index=True)

    df = df.fillna(0).astype(int)
    df.to_csv('jaeger_traces.csv', index=False)


# https://stackoverflow.com/questions/3160699/python-progress-bar
def show(j, count, prefix="", size=60,  out=sys.stdout):
    if j <= count:
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    else:
        print("\n", flush=True, file=out)


def get_traces(destination, end_time, total_requests, request_type = "compose", service = 'nginx-web-server', operation = '/wrk2-api/post/compose', jaeger = 'localhost'):
    traces_file = os.path.join(destination, f"{request_type}_traces.json")
    trace_ids_file = os.path.join(destination, f"{request_type}_trace_ids.txt")
    # Service name and desired operation
    
    experiment_duration = 2 * 24 * 60 * 60 * 1000 * 1000 # microseconds
    #TODO: remove?
    delta = 2 * 60 * 1000 * 1000 # 2 of delta minutes
    traces_per_lookup = 500
    n_collected_traces = 0
    #total_requests = 10000
    current_end_time = end_time
    lookback = (experiment_duration + delta) # you don't have to look back more than this
    current_start_time = end_time - lookback
    all_traces = []
    all_trace_ids = set()
    new_traces_added = True
    logging.info(f"Experiment folder: {destination}")
    while(n_collected_traces < total_requests) and new_traces_added:
        n_traces_this_lookup = min(traces_per_lookup, (total_requests - n_collected_traces))
        url_prefix = f"http://{jaeger}:16686/api/traces?end={current_end_time}&limit={n_traces_this_lookup}"
        url_suffix = f"&operation={operation}&service={service}&start={current_start_time}"
        url = str(url_prefix)+str(url_suffix)
    
        res = requests.get(url)
    
        jdata = json.loads(res.content.decode('utf-8'))
        traces = jdata['data']
        if len(traces) == 0:
            print("Stopping as the number of traces is 0")
            break
        next_end_time = current_end_time
        new_traces_added = False
        for trace in traces:
            if trace['traceID'] not in all_trace_ids:
                new_traces_added = True
                interestedIdx = 0 # this should ideally be the nginx span but good approx
                trace_start_approx = trace['spans'][interestedIdx]['startTime']
                next_end_time = min(trace_start_approx, next_end_time)
                all_trace_ids.add(trace['traceID'])
                all_traces.append(trace)
                n_collected_traces+=1
                diffFromStartTime = (trace_start_approx - current_start_time)/(1000*1000)
                #print ("\t req-num: %d traceID: %s curTraceStartTime: %s nextSetEndTime: %s diffFromStartTime: %.3f "%(n_collected_traces,trace['traceID'],trace_start_approx,next_end_time ,diffFromStartTime))
    
    
        current_end_time = next_end_time
        current_start_time = current_end_time - lookback
        time.sleep(0.1)
    
        deltaTS = current_end_time - current_start_time 
        deltaTS/=(1000*1000)
        diff = (current_end_time - current_start_time)/(1000*1000)
 
        show(n_collected_traces, total_requests, "Collecting traces:")

        #print ("\t New startTime: %d endTime: %d diff: %.3f deltaTS: %.3f "%(current_start_time, current_end_time,diff,deltaTS))
    
    logging.info(f"Total traces collected {len(all_traces)}")
    data = {"data":all_traces}
    with open(traces_file, "w") as f:
        json.dump(data, f)
    with open(trace_ids_file, "w") as f:
        f.write(",".join([str(id) for id in all_trace_ids]))
    
    #trace_data = extract_traces(service_name, traces)
    
    # Generate CSV
    #generate_csv(trace_data)