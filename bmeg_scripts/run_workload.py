import os
import signal
import sys
import subprocess
import yaml
import pathlib
import shutil
import logging
import time


from kubernetes import client, config, utils

import jaeger_fetch

logging.basicConfig(level=logging.INFO)


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


def static_workload(warm_up_duration, experiment_duration, destination_folder, frontend_cluster_ip, rps, port = "8080"):
        compose_rps = int(rps * 0.1)
        home_rps = int(rps * 0.6)
        user_rps = int(rps * 0.3)
        threads = 16
        connections = 32
        if warm_up_duration:
            subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/compose_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/compose-post.lua http://{frontend_cluster_ip}:{port}/wrk2-api/post/compose -R {compose_rps} > {destination_folder}/compose.warm_up &", shell=True, check = True)
            subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/home_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/read-home-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/home-timeline/read -R {compose_rps} > {destination_folder}/home.warm_up &", shell=True, check = True)
            subprocess.run(f"{wrk2_folder}/wrk2/wrk -D exp -t {threads} -c {connections} -d {warm_up_duration} -P {destination_folder}/user_latencies.warm_up -L -s {wrk2_folder}/wrk2/scripts/social-network/read-user-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/user-timeline/read -R {compose_rps} > {destination_folder}/user.warm_up &", shell=True, check = True)
            os.system(f"sleep {warm_up_duration}")

        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/compose_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/compose-post.lua http://{frontend_cluster_ip}:{port}/wrk2-api/post/compose -R {compose_rps} > {destination_folder}/compose.log &", shell=True, check = True)
        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/home_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/read-home-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/home-timeline/read -R {home_rps} > {destination_folder}/home.log &", shell=True, check = True)
        subprocess.run(
            f"{wrk2_folder}/wrk2/wrk -D exp -t {threads}  -c {connections} -d {experiment_duration} -P {destination_folder}/user_latencies.txt -L -s {wrk2_folder}/wrk2/scripts/social-network/read-user-timeline.lua http://{frontend_cluster_ip}:{port}/wrk2-api/user-timeline/read -R {user_rps} > {destination_folder}/user.log &", shell=True, check = True)
        os.system(f"sleep {experiment_duration}")


def get_pods(namespace):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    return v1.list_namespaced_pod(namespace)


def is_app_down():
    pass

def wait_for_state(namespace, state, sleep=30):
    pod_list = get_pods(namespace)
    while True:
        end = 1
        for pod in pod_list.items:
            if pod.status.phase != state:
                end = 0
        if end:
            break
        os.system(f"sleep {sleep}")

def is_deployment_successful(namespace = "social-network", failure_okay = ["write-home-timeline-service"] ):
    pod_list = get_pods(namespace)
    for pod in pod_list.items:
        name = pod.metadata.name
        #TODO: breaks if naming convention is changed - https://stackoverflow.com/questions/46204504/kubernetes-pod-naming-convention
        if name.rsplit('-',2)[0] not in failure_okay and pod.status.container_statuses[0].restart_count > 0: # just the deployment name after removing the replica set name and replica names
            return False
    return True


def get_n_requests(wrk2_log):
    p = subprocess.run(
        f"cat {wrk2_log} | grep 'requests in' | awk '{{print $1 }}'", stdout=subprocess.PIPE, shell=True
    )
    n_requests = p.stdout.decode("ascii").strip()
    return int(n_requests)  

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

connections = 32
threads = 16
namespace = "social-network"
request_types = ["compose", "home", "user"]
wrk2_folder = "/home/ubuntu/firm_compass"
experiment_folder = "/home/ubuntu/firm_compass/experiments"
rps_list = [600,700,800,900,1000]
rps_list = [800]
n_sequences = 10
worker_nodes = [f"userv{i}" for i in range(2,17)] # read from a config file
logging.info(f"Worker nodes {worker_nodes}")
experiment_duration = 1200
warm_up_duration = 300
seconds_to_microseconds = 1000 * 1000 
k8s_source = "/home/ubuntu/firm_compass/benchmarks/1-social-network/k8s-yaml-default"


for rps in rps_list:
    for sequence_number in range(n_sequences):
        clean_sn_app(k8s_source)
        clean_up_workers(worker_nodes)
        destination_folder = os.path.join(experiment_folder, f"test_{rps}_{sequence_number}")
        create_folder_p(destination_folder)
        k8s_destination = os.path.join(destination_folder, "k8s-yaml")
        create_folder_p(k8s_destination)
        place_pods("/home/ubuntu/firm_compass/benchmarks/1-social-network/placement/1.csv", k8s_source, k8s_destination)
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

        frontend_ip = get_service_cluster_ip("nginx-thrift", namespace)
        jaeger_ip = get_service_cluster_ip("jaeger-out", namespace)

        utilization_folder = os.path.join(destination_folder, "utilization_data")
        create_folder_p(utilization_folder)

        total_duration = warm_up_duration + experiment_duration
        utilization_reporting_interval = 30 # seconds
        subprocess.run(
            f"python3 /home/ubuntu/firm_compass/bmeg_scripts/kube_utilization.py {namespace} {total_duration} {utilization_reporting_interval} {utilization_folder} &",
                shell=True,
            )
        start = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Starting the workload at {start}")
        static_workload(warm_up_duration, experiment_duration, destination_folder, frontend_ip, rps)
        end = int(time.time() * seconds_to_microseconds) # epoch time in microseconds
        logging.info(f"Stopping the workload at {end}")
        jaeger_port_forward = subprocess.run("kubectl port-forward service/jaeger-out -n social-network --address 0.0.0.0 16686:16686 &", shell=True, capture_output=True) 
        os.system("sleep 5")
        for request in request_types:
            n_requests = get_n_requests(os.path.join(destination_folder, f"{request}.log"))
            jaeger_fetch.get_traces(destination_folder, end, n_requests, request_type=request, service = 'nginx-web-server', operation = f"/wrk2-api/post/{request}")
        os.kill(jaeger_port_forward.pid, signal.SIGTERM)