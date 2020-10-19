# Provision Kubernetes Cluster with the required Operators

This directory contains a Makefile thats downloads, patch, and install operators in the Kubernetes cluster.

## Problems encountered

While wiping the application on a K3S Cluster, the K8S namespace `trace-grapher` refuses to be deleted.
The command `kubectl delete -f operators/ECK/all-in-one.yaml` was struck, and the dashboard not responsive to manual deletions.
It turned out to be the some CRD that was not deleted properly and indefinitely wait for some `finalizers` to return [[source](https://github.com/kubernetes/kubernetes/issues/60538#issuecomment-369099998)].

> To manually remove the Kibana CRD finalizers that caused the issue on my app run `kubectl patch crd/kibanas.kibana.k8s.elastic.co -p '{"metadata":{"finalizers":[]}}' --type=merge`.

* * *
