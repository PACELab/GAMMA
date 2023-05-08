# Install docker
  sudo apt update;
  sudo apt install -qy docker.io;
  docker -v;

# Enable and start docker
  sudo systemctl enable docker;
  sudo systemctl status docker --no-pager;
  sudo systemctl start docker;

# Configure the Docker daemon, in particular to use systemd for the management of the container’s cgroups.
# https://kubernetes.io/docs/setup/production-environment/container-runtimes/#docker
# https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
  sudo mkdir -p /etc/docker;
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
  "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

  sudo systemctl enable docker && sudo systemctl daemon-reload && sudo systemctl restart docker;


# add kubernetes signing key
  curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add;

# add kube repository
  sudo apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main";

# install packages
# Don't pass "-y" flag as packages could already be installed and held at a particular version
  sudo apt install -qy kubeadm=1.24.2-00 kubelet=1.24.2-00 kubectl=1.24.2-00;


# avoids the package being automatically updated. Good for stability
  sudo apt-mark hold kubeadm kubelet kubectl;
  kubeadm version;

# turn off swap as kube doesn't work

  sudo swapoff -a

