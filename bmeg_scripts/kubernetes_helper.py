def get_pods(namespace):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    return v1.list_namespaced_pod(namespace)


def get_log_content(namespace, label_selector):
    config.load_kube_config()
    k8s_client = client.ApiClient()
    api_instance = client.CoreV1Api(k8s_client)
    pod_list = api_instance.list_namespaced_pod(namespace, label_selector=label_selector)
    content = api_instance.read_namespaced_pod_log(name=pod_list.items[0].metadata.name, namespace=namespace)
    return content