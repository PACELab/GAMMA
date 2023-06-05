import os
import sys

from kubernetes import client, config, utils


def get_metrics(namespace, total_duration, reporting_interval, destination):
    """
    Units of cpu = n -> nano CPUs
    Units of memory = Ki -> 1024 bytes
    """

    def return_key_and_stats_string(item, pod=False):
        nano_to_milli = 1000 * 1000
        Ki_to_bytes = 1024
        bytes_to_mb = 1000 * 1000
        row = []
        name = item["metadata"]["name"]
        row.append(item["timestamp"])
        row.append(item["window"])

        if pod:
            # list of containers in the pod. In our case, only one so hardcoding is fine.
            cpu_string = item["containers"][0]["usage"]["cpu"]
            # cpu usage can be zero in which case Kubernetes doesn't add units -___-
            cpu_in_nanocores = 0 if cpu_string == "0" else float(cpu_string[:-1])  # remove unit "n"
            memory_in_Ki = float(item["containers"][0]["usage"]["memory"][:-2])  # remove units "Ki"
        else:
            cpu_in_nanocores = float(item["usage"]["cpu"][:-1])  # remove unit "n"
            memory_in_Ki = float(item["usage"]["memory"][:-2])  # remove units "Ki"

        row.append(cpu_in_nanocores / nano_to_milli)
        row.append((memory_in_Ki * Ki_to_bytes) / bytes_to_mb)
        return name, ",".join(map(str, row))

    def write_files(dict, prefix):
        headers = ["timestamp", "window", "cpu (millicores)", "memory (megabytes)"]
        for key in dict:
            file = os.path.join(destination, prefix + key + ".csv")
            with open(file, "w") as f:
                f.write(",".join(headers))
                f.write("\n")
                f.write("\n".join(dict[key]))

    total_iterations = total_duration // reporting_interval  # extra logging
    config.load_kube_config()
    cust = client.CustomObjectsApi()
    # timestamp, window, usage["cpu"], usage["memory"]

    node_stats_collector = {}
    pod_stats_collector = {}

    for i in range(1, total_iterations + 1):
        node_list = cust.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")["items"]
        for node in node_list:
            name, stats_string = return_key_and_stats_string(node)
            # skip master
            if "master" in name:
                continue
            if name not in node_stats_collector:
                node_stats_collector[name] = []
            node_stats_collector[name].append(stats_string)

        pod_list = cust.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "pods")["items"]
        for pod in pod_list:
            if pod["metadata"]["namespace"] == namespace:
                name, stats_string = return_key_and_stats_string(pod, pod=True)
                if name not in pod_stats_collector:
                    pod_stats_collector[name] = []
                pod_stats_collector[name].append(stats_string)

        os.system(f"sleep {reporting_interval}")

    write_files(node_stats_collector, "node_")
    write_files(pod_stats_collector, "pod_")


if __name__ == "__main__":
    namespace = sys.argv[1]
    total_duration = int(sys.argv[2])  # in seconds
    reporting_interval = int(sys.argv[3])  # in seconds
    destination = sys.argv[4]
    get_metrics(namespace, total_duration, reporting_interval, destination)