# Overview

This repository contains a prototyped version of FIRM described in our [OSDI '20](https://www.usenix.org/conference/osdi20/presentation/qiu) paper "FIRM: An Intelligent Fine-grained Resource Management Framework for SLO-oriented Microservices".

## Instructions

### Machine Prerequisite

The following is the support matrix for Intel Cache Allocation Technology (CAT) and Memory Bandwidth Allocation (MBA) technology:

Intel(R) RDT hardware support

|Processor Types                                      | CMT | MBM | L3 CAT | L3 CDP | L2 CAT | MBA |
|-----------------------------------------------------|-----|-----|--------|--------|--------|-----|
|Intel(R) Xeon(R) processor E5 v3                     | Yes | No  | Yes    | No     | No     | No  |
|Intel(R) Xeon(R) processor D                         | Yes | Yes | Yes    | No     | No     | No  |
|Intel(R) Xeon(R) processor E3 v4                     | No  | No  | Yes    | No     | No     | No  |
|Intel(R) Xeon(R) processor E5 v4                     | Yes | Yes | Yes    | Yes    | No     | No  |
|Intel(R) Xeon(R) Scalable Processors                 | Yes | Yes | Yes    | Yes    | No     | Yes |
|Intel(R) Xeon(R) 2nd Generation Scalable Processors  | Yes | Yes | Yes    | Yes    | No     | Yes |
|Intel(R) Atom(R) processor for Server C3000          | No  | No  | No     | No     | Yes    | No  |

The ideal setting is "Intel(R) Xeon(R) Scalable Processors" or "Intel(R) Xeon(R) 2nd Generation Scalable Processors".
Check whether the machines in the cluster and the kernel version meet the requirements by executing:

```
lscpu | grep cat
lscpu | grep mba
uname -a
```

The minimum supported Linux kernel version is `4.12` and the following instructions are tested on Ubuntu 18.04.

### Setup Kubernetes Cluster

A running Kubernetes cluster is required before deploying FIRM. The following instructions are tested with Kubernetes v1.17.2, Docker v19.03.5, and Docker Compose v1.17.1. For set-up instructions, refer to [this](setup-k8s.md).

### Deploy FIRM

Clone the repository to the same location on every node.

Prerequisite: `pip3 install -r requirements.txt`

On each node, install anomaly injector:

```
cd anomaly-injector
make
cd sysbench
./autogen.sh
./configure
make -j
make install
```

Deploy tracing, metrics exporting, collection agents:

```
export NAMESPACE='monitoring'
kubectl create -f manifests/setup
kubectl create namespace observability
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/service_account.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/role.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/role_binding.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/operator.yaml
kubectl create -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/cluster_role.yaml
kubectl create -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/cluster_role_binding.yaml
kubectl create -f manifests/
```

Deploy graph database and pipeline for metrics storage (this requires docker-compose to be installed and enabled on the master node):

```
cd trace-grapher
docker-compose run stack-builder
# now a shell pops as root in the project directory of the stack-builder container
cd deploy-trace-grapher
make prepare-trace-grapher-namespace
make install-components
```

Install deployment module:

```
cd scripts
make all
cd python-cat-mba
make env
```

Check that all services and pods are up and running.

### Set Up Microservice Benchmarks With Tracing Enabled

#### Pre-requirements

- A running Kubernetes cluster with Docker and docker-compose installed on each node (as deployed above)
- Have the repository on all nodes (master and worker nodes) in the **same** path (otherwise many configuration paths need modification)
- Python 3.5+ (with `asyncio` and `aiohttp` installed)
- libssl-dev (`sudo apt install libssl-dev`)
- libz-dev (`sudo apt install libz-dev`)
- luarocks (`sudo apt install luarocks`)
- luasocket (`sudo luarocks install luasocket`)
- Make sure the following ports are available on each node: 8080, 8081, 16686

#### Configure the DNS Resolver

Set the resolver to be the FQDN of the core-dns or kube-dns service of your cluster in `<path-of-repo>/benchmarks/1-social-network/nginx-web-server/conf/nginx-k8s.conf` (line 44) and `<path-of-repo>/benchmarks/1-social-network/media-frontend/conf/nginx-k8s.conf` (line 37).

You can also use the cluster IP of your `kube-dns` service in `kube-system` namespace.

```
kubectl get service -n kube-system
```

#### Deploy Services

Change the `/root/firm/` to `<path-of-repo>` in `media-frontend.yaml` and `nginx-thrift.yaml` under the `<path-of-repo>/benchmarks/1-social-network/k8s-yaml` directory.

Run `kubectl apply -f <path-of-repo>/benchmarks/1-social-network/k8s-yaml/social-network-ns.yaml` to create the namespace.

Run `kubectl apply -f <path-of-repo>/benchmarks/1-social-network/k8s-yaml/` and wait `kubectl -n social-network get pod` to show all pods with status Running.

#### Setup Services

- Use `kubectl -n social-network get svc nginx-thrift` to get its cluster IP.
- Copy & paste the cluster IP at `<path-of-repo>/benchmarks/1-social-network/scripts/init_social_graph.py` line 74.
- Register users and construct social graph by running `python3 ./scripts/init_social_graph.py`.
    - This will initialize a social graph based on Reed98 Facebook Networks, with 962 users and 18.8K social graph edges.
    - This script should be executed under `<path-of-repo>/benchmarks/1-social-network/` directory.

### To Run

Make sure all pods in all namespaces are running without error or being evicted.

#### Anomaly Injection

Configure the machine IP address (external or internal, or hostname), username, password (you can keep password empty if password-less login is set up in the cluster), as well as the repo path (should be the same location) in `firm/anomaly-injector/injector.py`. For example:

```
nodes = [
        '10.1.0.11', '10.1.0.12', '10.1.0.13', '10.1.0.14',
        '10.1.0.21', '10.1.0.22', '10.1.0.23', '10.1.0.24',
]
username = 'ubuntu'
password = 'password'
location = '/root/firm/anomaly-injector/'
```

If usernames and passwords are different on different machines, they should be represented as lists in which the order should be the same as the list of nodes.

Then on each node, prepare files for generating disk I/O contention:

```
cd anomaly-injector
mkdir test-files
cd test-files
sysbench fileio --file-total-size=150G prepare
```

To run anomaly injection: `python3 injector.py`.

#### Workload Generation

Configure cluster IP (from `kubectl -n social-network get svc nginx-thrift`) at:
    - `<path-of-repo>/benchmarks/1-social-network/wrk2/scripts/social-network/compose-post.lua:66`;
    - `<path-of-repo>/benchmarks/1-social-network/wrk2/scripts/social-network/read-home-timeline.lua:16`;
    - `<path-of-repo>/benchmarks/1-social-network/wrk2/scripts/social-network/read-user-timeline.lua:16`;

Go to the directory of the HTTP workload generator: `cd <path-of-repo>/benchmarks/1-social-network/wrk2` and build the workload generator: `make`.

To run workload generation: `./wrk -D exp -t 8 -c 100 -R 1600 -d 1h -L -s ./scripts/social-network/compose-post.lua http://<clulster-ip>:8080/wrk2-api/post/compose`.
    - `t`: number of threads
    - `c`: number of connections
    - `R`: rate, i.e., number of requests per second
    - `d`: duration

#### SVM Training

Generate training dataset:

```
python3 metrics/analysis/cpa-training-labels.py &
python3 metrics/analysis/cpa-training-features.py &
```

Default training (customization on parameters see `minibatch_svm.py`):

```
python3 minibatch_svm.py
```

After training, checkpoints of the model will be automatically saved in the same directory. Inference in the future will base on these checkpoints.

#### RL Training

Start a redis server:

```
sudo systemctl start redis

# Install if needed: sudo apt install redis-server
```

Run metrics collector and store metrics in Redis:

```
python3 metrics/collector/collector.py

# Or, run with gunicore:
gunicore --worker=1 --log-level debug --log-file=- --bind 0.0.0.0:$COLLECTOR_PORT 'collector:build_app()'
# The default environment variable COLLECTOR_PORT is 8787
# Examples of other environment variable required:
COLLECTOR_REDIS_HOST=192.168.222.5 # IP of the host machine running redis server
COLLECTOR_REDIS_PORT=6379          # default
STATS_LEN=1440
```

Run sender which polls cAdvisor via its REST API:

```
# Replace /path/to/repository in metrics/sender/cron/crontab
# Replace the IP address of the machine running cAdvisor and collector
# Examples of environment variable required:
COLLECTOR_URL=http://192.168.222.5:8787/cadvisor/metrics/
CADVISOR_URL=http://192.168.222.5:8080/metrics/

crontab metrics/sender/crontab
```

Run server which returns states and accepts actions to execute (on each node): `python3 server.py`.
Run client which serves as the environment for RL (on master node): `python3 client.py`.
Start training (on master node): `python3 ddpg/main.py`.

#### Clean-up

On each node, clean up files used for generating disk I/O bandwidth contention.

```
cd anomaly-injector/test-files
sysbench fileio --file-total-size=150G cleanup
```

## Contact

Haoran Qiu - haoranq4@illinois.edu
