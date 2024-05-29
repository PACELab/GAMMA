from kubernetes import client, config
import datetime

config.load_kube_config()

apps_v1_api = client.AppsV1Api()
core_v1_api = client.CoreV1Api()


def restart_microservice(namespace, deployment_name):
    """Restarts a microservice by triggering a rolling update."""
    body = {'spec': {'template': {'metadata': {'annotations': {'kubectl.kubernetes.io/restartedAt': datetime.datetime.utcnow().isoformat()}}}}}
    apps_v1_api.patch_namespaced_deployment(deployment_name, namespace, body)

def redeploy_microservice(namespace, deployment_name, new_image):
    """Redeploys a microservice with a new Docker image."""
    deployment = apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
    deployment.spec.template.spec.containers[0].image = new_image  # Update container image
    apps_v1_api.patch_namespaced_deployment(deployment_name, namespace, deployment)

def scale_up_microservice(namespace, deployment_name, desired_replicas):
    """Scales up a microservice to the desired number of replicas."""
    body = {'spec': {'replicas': desired_replicas}}  
    apps_v1_api.patch_namespaced_deployment_scale(deployment_name, namespace, body)

def scale_down_microservice(namespace, deployment_name, desired_replicas):
    """Scales down a microservice to the desired number of replicas."""
    if desired_replicas < 1:
        raise ValueError("Replicas count cannot be less than 1")
    scale_up_microservice(namespace, deployment_name, desired_replicas)



namespace = "default"  
deployment_name = "nginx-deployment" 
new_image = "nginx:1.25.5"
desired_replicas = 3


# restart_microservice(namespace, deployment_name)
# redeploy_microservice(namespace, deployment_name, new_image)
# scale_up_microservice(namespace, deployment_name, desired_replicas)
scale_down_microservice(namespace, deployment_name, desired_replicas - 1)
