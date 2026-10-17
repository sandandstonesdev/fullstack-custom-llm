# OpenTofu root module - STUB, filled in once the cluster target is chosen.
#
# Planned layout (keep names stable so CI can rely on them):
#   deploy/tofu/main.tf - variables + local state (small project, no remote backend yet)
#   deploy/tofu/k8s.tf  - cluster resource + kubernetes provider wired to it
#
# Apply with:
#   cd deploy/tofu && tofu apply

variable "cluster_name" {
  description = "Name of the k8s cluster that serves the app"
  type        = string
  default     = "polish-llm"
}

# TODO: pick provider (managed k8s) and add the cluster resource in k8s.tf.
