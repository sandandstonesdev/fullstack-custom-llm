# Kubernetes cluster + provider wiring - STUB.
#
# TODO once the provider is chosen:
#   resource "<provider>_cluster" "main" { ... }
#   provider "kubernetes" {
#     host  = <provider>_cluster.main.endpoint
#     token = <provider>_cluster.main.token  # or client certs
#   }
#
# After `tofu apply`, manifests are applied manually for now:
#   kubectl apply -f deploy/k8s/
