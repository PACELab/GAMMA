import os
import signal
import sys
import subprocess
import yaml
import pathlib
import shutil
import logging
import time
import requests
import json
import threading


from kubernetes import client, config, utils

import jaeger_fetch
import arguments
import create_dataset

logging.basicConfig(
    #filename='HISTORYlistener.log',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class Bottlenecks:
    def __init__(self, node, measure_list, duration_list):
        self.node = node
        self.measure_list = measure_list
        self.duration_list = duration_list



def create_folder_p(folder):
    """
    mkdir -p (create parent directories if necessary. Don't throw error if directory already exists)
    """
    pathlib.Path(f"{folder}").mkdir(parents=True, exist_ok=True)

def write_manifest_file(manifest_file, data=""):
    """
    Write a manifest file. If data is not provided, creates an empty file.
    """
    with open(manifest_file, "w") as f:
        yaml.dump_all(data, f, default_flow_style=False)

def load_multi_doc_yaml(manifest_file):
    """
    Load a manifest file.
    """
    with open(manifest_file) as f:
        data = list(yaml.load_all(f, Loader=yaml.FullLoader))

    return data

def place_pods(placement_file, source, destination):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    with open(placement_file) as f:
        for line in f:
            service, node = [i.strip() for i in line.split(',')]
            logging.debug(f"placing {service} on {node}")
            template_file = os.path.join(destination, f"{service}.yaml")
            data = load_multi_doc_yaml(template_file)
            deployment_doc_index = 2 if service == "jaeger" else  1
            data[deployment_doc_index]["spec"]["template"]["spec"]["nodeSelector"]= {"kubernetes.io/hostname": node}
            write_manifest_file(template_file, data)


def get_service_cluster_ip(service, namespace):
    p = subprocess.run(
        f"kubectl get svc {service} -n {namespace} -ojsonpath='{{.spec.clusterIP}}'", stdout=subprocess.PIPE, shell=True
    )
    cluster_ip = p.stdout.decode("ascii").strip()
    return cluster_ip


def clean_up_workers(worker_nodes):
    for node in worker_nodes:
        logging.info(f"Cleaning worker {node}.")
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm/benchmarks/1-social-network/tmp/*"')
        subprocess.run(
            f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm_compass/benchmarks/1-social-network/tmp/*"', shell=True, check=True
        )

def clean_sn_app(k8s_yaml_folder):
    logging.info(f"Deleting the application.")
    subprocess.run(f"kubectl delete -f {k8s_yaml_folder}", shell=True)
    os.system("sleep 120")

def set_up_workers(worker_nodes):
    for node in worker_nodes:
        # tmp in cleaned up but just in case.
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm/benchmarks/1-social-network/tmp/*"')
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "cp -r /home/ubuntu/firm/benchmarks/1-social-network/volumes/* /home/ubuntu/firm/benchmarks/1-social-network/tmp/"')
        logging.info(f"Setting up worker node {node}")
        subprocess.run(
            f'ssh  -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm_compass/benchmarks/1-social-network/tmp/*"', shell=True, check=True)
        subprocess.run(
            f'ssh  -i /home/ubuntu/compass.key ubuntu@{node} "cp -r /home/ubuntu/firm_compass/benchmarks/1-social-network/volumes/* /home/ubuntu/firm_compass/benchmarks/1-social-network/tmp/"',  shell=True, check=True)

def set_up_sn(k8s_folder):
    logging.info("Deploying the application.")
    ns = os.path.join(k8s_folder, "social-network-ns.yaml")
    subprocess.run(f"kubectl apply -f {ns}", shell=True, check=True)
    subprocess.run(f"kubectl apply -f {k8s_folder}", shell=True, check=True)
    os.system("sleep 120")

def get_wrk2_params(request_type, app):
    """
    Customize for app and request_type
    """
    return 16, 32

def get_request_composition(app, rps):
    """
    have a dictionary for each app
    """
    logging.debug(f"Request compositions: compose - {int(rps * 0.1)}, home - {int(rps * 0.6)}, user - {int(rps * 0.3)}")
    return int(rps * 0.1), int(rps * 0.6), int(rps * 0.3)

def static_workload(experiment_duration, destination_folder, frontend_cluster_ip, rps, port = "8080"):
        compose_rps, home_rps, user_rps = get_request_composition(app, rps)
        threads, connections = get_wrk2_params("", app)

        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/compose_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/compose-post.lua http://{frontend_cluster_ip}:{port}/wrk2-api/post/compose -R {compose_rps} > {destination_folder}/compose.log &", shell=True, check = True)
        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/home_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/read-home-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/home-timeline/read -R {home_rps} > {destination_folder}/home.log &", shell=True, check = True)
        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/user_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/read-user-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/user-timeline/read -R {user_rps} > {destination_folder}/user.log &", shell=True, check = True)
        os.system(f"sleep {experiment_duration}")

def warm_up_app(warm_up_duration, destination_folder, frontend_cluster_ip, rps,  port = "8080"):
    compose_rps, home_rps, user_rps = get_request_composition(app, rps)
    threads, connections = get_wrk2_params("", app)
    subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/compose_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/compose-post.lua http://{frontend_cluster_ip}:{port}/wrk2-api/post/compose -R {compose_rps} > {destination_folder}/compose.warm_up &", shell=True, check = True)
    subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/home_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/read-home-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/home-timeline/read -R {home_rps} > {destination_folder}/home.warm_up &", shell=True, check = True)
    subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/user_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/read-user-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/user-timeline/read -R {user_rps} > {destination_folder}/user.warm_up &", shell=True, check = True)
    os.system(f"sleep {warm_up_duration}")


def get_pods(namespace):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    return v1.list_namespaced_pod(namespace)


def get_nodes():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    return v1.list_node()

def is_app_down():
    pass

def wait_for_state(namespace, state, sleep=30, max_wait=120):
    pod_list = get_pods(namespace)
    wait = 0
    while wait < max_wait:
        if all(pod.status.phase == state for pod in pod_list.items):
            break
        logging.info(f"Sleeping for {sleep}s waiting for the app to reach state {state}")
        os.system(f"sleep {sleep}")
        wait += sleep
    else:
        logging.error(f"App didn't reach the expected state: {state}")
        raise

def is_deployment_successful(namespace = "social-network", failure_okay = ["write-home-timeline-service"] ):
    pod_list = get_pods(namespace)
    logging.info("Checking if deployment was successful...")
    wait_for_state(namespace, "Running") 
    for pod in pod_list.items:
        name = pod.metadata.name
        #TODO: breaks if naming convention is changed - https://stackoverflow.com/questions/46204504/kubernetes-pod-naming-convention
        if name.rsplit('-',2)[0] not in failure_okay and pod.status.container_statuses[0].restart_count > 0: # just the deployment name after removing the replica set name and replica names
            logging.error("App deployment failed :/")
            return False
    logging.info("Deployment successful!")
    return True


def get_io_bottleneck_cmd(measure, duration):
    """
       -d N, --hdd N
              start  N  workers  continually  writing,  reading and removing temporary files. The
              default mode is to stress test sequential writes and reads.  With the  --aggressive
              option  enabled  without  any --hdd-opts options the hdd stressor will work through
              all the --hdd-opt options one by one to cover a range of I/O options.
    """
    command = f"stress-ng --aggressive --hdd {measure} -t {duration}s"
    return command

def get_network_bottleneck_cmd(measure, duration):
    #command = f"stress-ng --net-delay {measure}ms --timeout {duration}"
    """
        -S N, --sock N
              start  N  workers that perform various socket stress activity. This involves a pair
              of  client/server  processes  performing  rapid  connect,  send  and  receives  and
              disconnects on the local host.   
    """
    command  = f"stress-ng --sock {measure} -t {duration}s"
    return command

def get_memory_bottleneck_cmd(measure, duration):
    """
    https://unix.stackexchange.com/questions/99334/how-to-fill-90-of-the-free-memory
    """
    command = command = f"stress-ng --vm-bytes $(awk '/MemAvailable/{{printf \"%d\\n\", $2 * {measure};}}' < /proc/meminfo)k --vm-keep -m 1 --timeout {duration}"
    return command

def get_cpu_bottleneck_cmd(measure, duration):
    command = f"python3 /home/ubuntu/firm_compass/tools/CPULoadGenerator/CPULoadGenerator.py -l {measure} -d {duration} -c 0 -c 1 -c 2 -c 3"
    return command

def create_bottlenecks_remotely(bottleneck, destination, bottleneck_type):
    """
    duration_list should have periods of non-bottleneck and bottleneck phases.
    """
    # TODO: assert that the sum of phases is greater than the experiment duration and the extra time.
    get_cmd_function = {"cpu": get_cpu_bottleneck_cmd, 
                     "memory": get_memory_bottleneck_cmd,
                     "network" : get_network_bottleneck_cmd,
                     "io" : get_io_bottleneck_cmd,}[bottleneck_type]
    grace_period = 1 # in seconds
    current_thread = threading.current_thread().name
    logging.debug(f"Thread {current_thread} running on {bottleneck.node}")
    try:
        with open(os.path.join(destination, f"{bottleneck_type}_{bottleneck.node}_phases"), "w") as f:
            for i in range(len(bottleneck.duration_list)):
                if i%2 ==0:
                    logging.debug(f"{current_thread} sleeping for {bottleneck.duration_list[i]}")
                    os.system(f"sleep {bottleneck.duration_list[i]}")
                else:
                    logging.debug(f"{current_thread} creating bottleneck with measure {bottleneck.measure_list[i//2]/100} for {bottleneck.duration_list[i]}")
                    f.write(f"Bottleneck of type {bottleneck_type} with measure {bottleneck.measure_list[i//2]/100} starts at {time.time()}\n")
                    # load percentage should be [0.1]
                    command = get_cmd_function(bottleneck.measure_list[i//2]/100, bottleneck.duration_list[i])
                    p = subprocess.Popen(f'ssh -i /home/ubuntu/compass.key ubuntu@{bottleneck.node} "{command}"', shell=True)
                    try:
                        # the CPU load generator doesn't terminate.
                        _, _ = p.communicate(timeout=bottleneck.duration_list[i] + grace_period)
                    except subprocess.TimeoutExpired:
                        logging.error(f"{current_thread} is stuck.")
                        p.kill()
                    f.write(f"Bottleneck of type {bottleneck_type} with measure {bottleneck.measure_list[i//2]/100} ends at {time.time()}\n")
        logging.debug(f"{current_thread} is terminating...")
    except Exception as e:
        logging.exception(f"{current_thread} Exception occurred in thread for {bottleneck.node}: {e}")


def create_bottlenecks(bottlenecked_nodes, interference_percentage, phases, experiment_folder, bottleneck_type):
    threads = []
    for i, node in enumerate(bottlenecked_nodes):
        threads.append(threading.Thread(target=create_bottlenecks_remotely, args=(Bottlenecks(node,interference_percentage, phases), experiment_folder, bottleneck_type)))
        logging.debug(f"Starting thread on {node}")
        threads[i].start()
    return threads

def create_and_setup_experiment_folder(args, experiments_root, rps, sequence_number):
    destination_folder = os.path.join(experiments_root, f"{args.experiment_name}_{rps}_{sequence_number}")
    create_folder_p(destination_folder)
    with open(os.path.join(destination_folder, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return destination_folder

def deploy_application(destination_folder, placement_config,  worker_nodes, k8s_source, rps, sequence_number):
    clean_sn_app(k8s_source)
    clean_up_workers(worker_nodes)
    create_folder_p(destination_folder)
    k8s_destination = os.path.join(destination_folder, "k8s-yaml")
    create_folder_p(k8s_destination)
    place_pods(placement_config, k8s_source, k8s_destination)
    set_up_workers(worker_nodes)
    set_up_sn(k8s_destination)
    max_retries = 3
    retries = 0
    while (not is_deployment_successful()) and (retries < max_retries):
        retries += 1
        clean_sn_app(k8s_source)
        clean_up_workers(worker_nodes)
        set_up_workers(worker_nodes)
        set_up_sn(k8s_destination)
    return destination_folder

def get_n_requests(wrk2_log):
    p = subprocess.run(
        f"cat {wrk2_log} | grep 'requests in' | awk '{{print $1 }}'", stdout=subprocess.PIPE, shell=True
    )
    n_requests = p.stdout.decode("ascii").strip()
    return int(n_requests)  


def get_service_name(request_type):
    service_name_lookup = {
        # SN app
        'compose': 'nginx-web-server',
        'home' :  'nginx-web-server',
        'user' : 'nginx-web-server' ,
    
    
        # MM app
        'reviewCompose' : 'nginx',
        'reviewRead' : 'nginx',
        'plot' : 'nginx',
    
        # TT app
        'book': 'ts-preserve-other-service',
        'search': 'ts-travel-service',
    }
    return service_name_lookup[request_type]

def get_operation_name(request_type):
    operation_name_lookup = {
            #SN app
            'compose' : '/wrk2-api/post/compose',
            'home' : '/wrk2-api/home-timeline/read',
            'user' : '/wrk2-api/user-timeline/read',
            #MM app
        'reviewCompose' : '/wrk2-api/review/compose',
        'reviewRead' : '/wrk2-api/review/read',
        'plot' : '/wrk2-api/plot/read',
        #TT app
        'search': 'queryInfo', 
        'book': 'preserve',
            }
    return operation_name_lookup[request_type]

def subprocess_bg(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, start_new_session=True)

def subprocess_kill_bg(process):
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def get_jaeger_traces(request_types, end, destination_folder, namespace = "social-network", forwarding_port=16686, service_port=16686, jaeger_service_name="jaeger-out"):
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    jaeger_port_forward = subprocess_bg(f"kubectl port-forward service/{jaeger_service_name} -n {namespace} --address 0.0.0.0 {forwarding_port}:{service_port}")
    os.system("sleep 5")
    try:
        for request in request_types:
            n_requests = get_n_requests(os.path.join(destination_folder, f"{request}.log"))
            jaeger_fetch.get_traces(destination_folder, end, n_requests, request_type=request, service = get_service_name(request), operation = get_operation_name(request))
        status = True
    except Exception as e:
        logging.error(f"Unable to download jaeger traces: {e}")
        status = False
    finally:
        # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
        # since shell=True
        subprocess_kill_bg(jaeger_port_forward)
        return status


def save_logs(pod_list, namespace, destination):
    for pod in pod_list.items:
        pod_name = pod.metadata.name
        # save both output and errors
        logging.info(f"Saving logs of pod: {pod_name}")
        os.system(f"kubectl logs {pod_name} -n {namespace} > {destination}/{pod_name}.log 2>&1")
        # get the previously terminated container's log if it exists
        os.system(
            f"kubectl logs {pod_name} -n {namespace} --previous > {destination}/{pod_name}_previous.log 2>&1"
        )

def save_pods_describe(pod_list, namespace, destination ):
    for pod in pod_list.items:
        # save both output and errors
        pod_name = pod.metadata.name
        logging.info(f"Saving describe output of pod: {pod_name}")
        os.system(f"kubectl describe pods {pod_name} -n {namespace} > {destination}/{pod_name}.log 2>&1")

def save_nodes_describe(node_list, destination ):
    for node in node_list.items:
        node_name = node.metadata.name
        # save both output and errors
        logging.info(f"Saving describe output of node: {node_name}")
        os.system(f"kubectl describe nodes {node_name} > {destination}/{node_name}.log 2>&1")

def query_for_prom_metrics(metric, instance):
    """
    memory_utilization: (total-available)/total
    """
    instance = "your_instance_value"  # Replace with the actual instance value
    
    metrics = {
        "memory_utilization": f'100*(node_memory_MemTotal_bytes{{job="node-exporter", instance="{instance}"}} - '
                               f'node_memory_MemAvailable_bytes{{job="node-exporter", instance="{instance}"}}) / '
                               f'node_memory_MemTotal_bytes{{job="node-exporter", instance="{instance}"}}',
        "node_cpu_seconds_total": ""
    }



def get_prometheus_node_metrics(node_list, node_exporter_namespace="monitoring"):
    metrics = []
    monitoring_pod_list = get_pods("monitoring")
    for pod in monitoring_pod_list:
        name = pod.metadata.name
        if "node-exporter" in name:
            ip = pod.status.pod_ip
            for metric in metrics:
                pass

def get_prometheus_pod_metrics(pod_list, namespace, experiment_folder, folder =  "prom_metrics", server_url="localhost", port = 9200, duration=1800):
    metrics = ["container_cpu_usage_seconds_total", 
    "container_memory_usage_bytes", 
    "container_memory_cache"
    "container_memory_failcnt",
    "container_memory_failures_total",
    "container_oom_events_total",
    "container_network_receive_bytes_total", 
    "container_network_receive_errors_total",
    "container_network_receive_packets_dropped_total",
    "container_network_transmit_bytes_total",
    "container_fs_writes_bytes_total",
    "container_fs_reads_bytes_total",
    "container_llc_occupancy_bytes",
    "container_processes",
    "container_sockets",
    "container_threads",
    ]

    metrics_with_no_container_label = ["container_network_receive_bytes_total", 
    "container_network_transmit_bytes_total",
    ]
    prom_metrics_folder = os.path.join( experiment_folder, folder)
    create_folder_p(prom_metrics_folder)   
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    prometheus_port_forward = subprocess_bg(f"kubectl port-forward service/prometheus-service -n monitoring --address 0.0.0.0 9200:8080")
    os.system("sleep 5")
    try:
        for pod in pod_list.items:
            for metric in metrics:
                container = pod.spec.containers[0].name
                pod_name = pod.metadata.name
                #logging.debug(f"Saving metric {metric} of container: {container}")
                if metric in metrics_with_no_container_label:
                    query = f'{metric}{{namespace="{namespace}", pod="{pod_name}"}}[{duration}s]'
                else:
                    query = f'{metric}{{namespace="{namespace}", container="{container}", pod="{pod_name}"}}[{duration}s]'                
                response =requests.get(f"http://{server_url}:{port}" + '/api/v1/query', params={'query': query})
                try:
                    with open(os.path.join(prom_metrics_folder, f"{container}_{metric}"), "w") as f:
                        json.dump(response.json()['data']['result'][0]["values"], f, indent=4)
                except Exception as e:
                    logging.error(f"Unable to get {metric} for {container} due to exception: {e}")
    finally:
        subprocess_kill_bg(prometheus_port_forward)

def get_pod_logs(namespace, experiment_folder, pod_list, folder):
    pods_logs_destination = os.path.join(experiment_folder, folder)
    create_folder_p(pods_logs_destination)
    save_logs(pod_list, namespace, pods_logs_destination)

def get_pod_describe_logs(namespace, experiment_folder, app_pod_list):
    pods_describe_destination = os.path.join(experiment_folder, "pods_describe")
    create_folder_p(pods_describe_destination)
    save_pods_describe(app_pod_list, namespace, pods_describe_destination )


def get_nodes_describe(node_list , experiment_folder):
    nodes_describe_folder = os.path.join( experiment_folder, "nodes_describe")
    create_folder_p(nodes_describe_folder)
    save_nodes_describe(node_list, nodes_describe_folder )

def get_logs_and_metrics(namespace, experiment_folder):
    app_pod_list = get_pods(namespace)
    kube_pod_list = get_pods( "kube-system")
    node_list = get_nodes()
    
    get_pod_logs(namespace, experiment_folder, app_pod_list, folder="app_pod_logs")
    get_pod_logs("kube-system", experiment_folder, kube_pod_list, folder="kube_pod_logs")
    get_pod_describe_logs(namespace, experiment_folder, app_pod_list, ) 
    get_nodes_describe(node_list, experiment_folder)     
    get_prometheus_pod_metrics(app_pod_list, namespace, experiment_folder)

def start_metric_collecter(namespace, experiment_duration, warm_up_duration, experiment_folder):
    utilization_folder = os.path.join(experiment_folder, "utilization_data")
    create_folder_p(utilization_folder)
    total_duration = warm_up_duration + experiment_duration
    utilization_reporting_interval = 30 # seconds
    subprocess.run(
            f"python3 /home/ubuntu/firm_compass/bmeg_scripts/kube_utilization.py {namespace} {total_duration} {utilization_reporting_interval} {utilization_folder} &",
                shell=True,
            )


#prom_conts
prom_rate_duration = "3s"

args = arguments.argument_parser()
app = "SN"
connections = 32
threads = 16
namespace = "social-network"
request_types = ["compose", "home", "user"]
wrk2_folder = "/home/ubuntu/firm_compass"
experiments_root = "/mnt/experiments"
rps_list = args.rps
starting_sequence = args.starting_sequence
n_sequences = 30
worker_nodes = [f"userv{i}" for i in range(2,17)] # read from a config file
logging.info(f"Worker nodes {worker_nodes}")
experiment_duration = args.experiment_duration
warm_up_duration = args.warm_up_duration
setup_duration = 600
seconds_to_microseconds = 1000 * 1000 
k8s_source = "/home/ubuntu/firm_compass/benchmarks/1-social-network/k8s-yaml-default"
placement_config_version = 1
placement_config = f"/home/ubuntu/firm_compass/benchmarks/1-social-network/placement/{placement_config_version}.csv"


# experiment_folder = "/mnt/experiments/realistic_july29_800_0"
# end = 1690665922773286
# if get_jaeger_traces(request_types, end, experiment_folder):
#             create_dataset.main(app, experiment_folder, 0)
# sys.exit()

for rps in rps_list:
    for sequence_number in range(starting_sequence, n_sequences):
        experiment_folder = create_and_setup_experiment_folder(args, experiments_root, rps, sequence_number)
        deploy_application(experiment_folder, placement_config, worker_nodes, k8s_source, rps, sequence_number)

        frontend_ip = get_service_cluster_ip("nginx-thrift", namespace)
        jaeger_ip = get_service_cluster_ip("jaeger-out", namespace)

        # start_metric_collecter(namespace, experiment_duration, warm_up_duration, experiment_folder)

        start = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Starting warm-up workload at {start}")
        warm_up_app(warm_up_duration, experiment_folder, frontend_ip, rps)
        end = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Stopping the warm-up workload at {end}")

        threads = []
        if args.cpu_bottlenecked_nodes is not None:
            logging.info("Creating bottlenecks")
            threads += create_bottlenecks(args.cpu_bottlenecked_nodes, args.cpu_interference_percentage, args.cpu_phases, experiment_folder, "cpu")
        if args.mem_bottlenecked_nodes is not None:
            logging.info("Creating bottlenecks")
            threads += create_bottlenecks(args.mem_bottlenecked_nodes, args.mem_interference_percentage, args.mem_phases, experiment_folder, "memory")
        if args.net_bottlenecked_nodes is not None:
            logging.info("Creating bottlenecks")
            threads += create_bottlenecks(args.net_bottlenecked_nodes, args.net_interference_percentage, args.net_phases, experiment_folder, "memory")
        if args.io_bottlenecked_nodes is not None:
            logging.info("Creating bottlenecks")
            threads += create_bottlenecks(args.io_bottlenecked_nodes, args.io_interference_percentage, args.io_phases, experiment_folder, "memory")

        start = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Starting the workload at {start}")
        static_workload(experiment_duration, experiment_folder, frontend_ip, rps)
        end = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Stopping the workload at {end}")
        
        if not args.skip_log_metric_collection:
            get_logs_and_metrics(namespace, experiment_folder)
        if not args.skip_trace_collection and get_jaeger_traces(request_types, end, experiment_folder):
            create_dataset.main(app, experiment_folder, 0)
        for thread in threads:
            logging.debug(f"is thread alive : {thread.is_alive()}")
            thread.join()

        # write the list of <bottlencks, source of bottlenecks, and metadata for source>
        #write_bottlenecks(bottleneck_file, graph_pathszz)