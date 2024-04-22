import pathlib
import os
import subprocess
import shutil

def create_folder_p(folder):
    """
    mkdir -p (create parent directories if necessary. Don't throw error if directory already exists)
    """
    pathlib.Path(f"{folder}").mkdir(parents=True, exist_ok=True)

def scp_helper(
    source_path,
    destination_path,
    source_user=None,
    destination_user=None,
    source_host=None,
    destination_host=None,
    key_file=None,
):
    command = "scp"
    if key_file is None:
        if source_host is None:
            command += f" -r {source_path} {destination_user}@{destination_host}:{destination_path}"
        else:
            command = f"scp {source_user}@{source_host}:{source_path} {destination_user}@{destination_host}:{destination_path}"
    else:
        command = f"scp -i {key_file} {source_user}@{source_host}:{source_path} {destination_user}@{destination_host}:{destination_path}"
    os.system(command)


def ssh_handler(user, host, use_identity_file=True, key="/home/azureuser/gagan-aiopsbench_key.pem", cmd_on_remote=""):
    ssh_command = "ssh "

    if use_identity_file:
        ssh_command += f"-i {key} "
    ssh_command += f"{user}@{host} "

    if cmd_on_remote:
        ssh_command += f'"{cmd_on_remote}"'

    return run_command(ssh_command, return_output=True)


def run_command(command, return_output):
    try:
        if return_output:
            p = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
            return p.stdout.decode("ascii").strip()
        else:
            p = subprocess.run(command, shell=True)
    except Exception as e:
        print(e)
        raise e
    
def copy_scripts(repo_parent_path, experiment_version_folder):
    destination = os.path.join(experiment_version_folder, "tools")
    source = os.path.join(repo_parent_path, "cross-layer-tuning", "tools")
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def save_logs(destination, namespace):
    save_logs_hardlinks(destination, namespace)


def save_logs_helper(namespace, destination, folder_name):
    log_folder = os.path.join(destination, folder_name)
    # same as app_log_folder but avoids confusion
    compressed_file = os.path.join(destination, folder_name)
    create_folder_p(log_folder)
    get_logs(namespace, log_folder)
    # https://docs.python.org/3/library/shutil.html#shutil-archiving-example-with-basedir
    # https://stackoverflow.com/questions/32640053/compressing-directory-using-shutil-make-archive-while-preserving-directory-str
    shutil.make_archive(compressed_file, "gztar", destination, folder_name)
    shutil.rmtree(log_folder)


def save_logs_hardlinks(destination, namespace):
    save_logs_helper("social-network", destination, "pod_logs")
    save_logs_helper("kube-system", destination, "kube_logs")


def get_logs(namespace, destination):
    pod_list = kubernetes_helper.get_pods(namespace)
    for pod in pod_list.items:
        os.system(f"kubectl logs {pod.metadata.name} -n {namespace} > {destination}/{pod.metadata.name}.log")
        # get the previously terminated container's log if it exists
        os.system(
            f"kubectl logs {pod.metadata.name} -n {namespace} --previous > {destination}/{pod.metadata.name}_previous.log 2>&1"
        )
