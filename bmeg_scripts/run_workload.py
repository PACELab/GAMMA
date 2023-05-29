import os
import sys
import subprocess
import helper


def get_service_cluster_ip(service, namespace):
    p = subprocess.run(
        f"kubectl get svc {service} -n {namespace} -ojsonpath='{{.spec.clusterIP}}'", stdout=subprocess.PIPE, shell=True
    )
    cluster_ip = p.stdout.decode("ascii").strip()
    return cluster_ip


def clean_up_workers(worker_nodes):
    for node in worker_nodes:
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm/benchmarks/1-social-network/tmp/*"')
        subprocess.run(
            f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm_compass/benchmarks/1-social-network/tmp/*"', shell=True, check=True
        )

def set_up_workers(worker_nodes):
    for node in worker_nodes:
        # tmp in cleaned up but just in case.
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm/benchmarks/1-social-network/tmp/*"')
        # os.system(f'ssh -i /home/ubuntu/compass.key ubuntu@{node} "cp -r /home/ubuntu/firm/benchmarks/1-social-network/volumes/* /home/ubuntu/firm/benchmarks/1-social-network/tmp/"')
        subprocess.run(
            f'ssh  -i /home/ubuntu/compass.key ubuntu@{node} "sudo rm -rf /home/ubuntu/firm_compass/benchmarks/1-social-network/tmp/*"', shell=True, check=True)
        subprocess.run(
            f'ssh  -i /home/ubuntu/compass.key ubuntu@{node} "cp -r /home/ubuntu/firm_compass/benchmarks/1-social-network/volumes/* /home/onecogsselftuneadmin/cross-layer-tuning/benchmarks/1-social-network/tmp/"',  shell=True, check=True)

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

connections = 32
threads = 16
namespace = "social-network"
wrk2_folder = "/home/ubuntu/firm_compass"
rps_list = [800]
n_sequences = 10
worker_nodes = [f"userv{i}" for i in range(2,17)] # read from a config file
experiment_duration = 
frontend_ip = get_service_cluster_ip("nginx-thrift", namespace)
jaeger_ip = get_service_cluster_ip("jaeger-out", namespace)
for rps in rps_list:
    for sequence_number in n_sequences:
        clean_up_workers(worker_nodes)
        set_up_workers(worker_nodes)
        static_workload()
