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

# Jaeger API endpoint
jaeger_api_url = 'http://localhost:16686'

# Service name and desired operation
service_name = 'nginx-web-server'
operation_name = '/wrk2-api/post/compose'

# Retrieve traces from Jaeger API
url = f'{jaeger_api_url}/api/traces?service={service_name}&operation={operation_name}&limit=100000'
response = requests.get(url).json()
traces = response['data']

page = 1
page_size = 100
has_more_traces = True

while has_more_traces:
    url = f'{jaeger_api_url}/api/traces?service={service_name}&operation={operation_name}&limit={page_size}&offset={(page - 1) * page_size}'
    response = requests.get(url)
    data = response.json()
    traces.extend(data['data'])
    has_more_traces = len(data['data']) == page_size
    page += 1
print(len(traces))
# Extract trace data
#trace_data = extract_traces(service_name, traces)

# Generate CSV
#generate_csv(trace_data)
