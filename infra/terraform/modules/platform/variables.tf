variable "environment" {
  description = "Environment name (staging, prod)."
  type        = string
}

variable "zone_ids" {
  description = "Availability-zone IDs to use (pinned so the subnet layout never shifts when AWS adds a zone)."
  type        = list(string)
  default     = ["use1-az1", "use1-az2", "use1-az4"]
}

variable "vpc_cidr" {
  description = "VPC CIDR (/16 recommended)."
  type        = string
}

variable "single_nat_gateway" {
  description = "One NAT gateway instead of one per AZ (cheaper, less available)."
  type        = bool
}

variable "caller_cidrs" {
  description = "CIDRs of the ad servers allowed to call the API."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for the internal ALB."
  type        = string
}

variable "api_image_tag" {
  description = "API image tag (immutable, set by CD)."
  type        = string
}

variable "train_image_tag" {
  description = "Training image tag (immutable, set by CD)."
  type        = string
}

variable "api_min_tasks" {
  description = "Minimum API tasks."
  type        = number
}

variable "api_max_tasks" {
  description = "Maximum API tasks."
  type        = number
}

variable "redis_node_type" {
  description = "ElastiCache node type."
  type        = string
}

variable "redis_replicas" {
  description = "Redis read replicas."
  type        = number
}

variable "data_retention_days" {
  description = "Days to keep exports and decision logs."
  type        = number
  default     = 400
}

variable "deletion_protection" {
  description = "Protect stateful resources from deletion."
  type        = bool
}

variable "github_repository" {
  description = "owner/name of the repository that deploys."
  type        = string
  default     = "lucasbabur/charade"
}

variable "create_github_oidc_provider" {
  description = "Create the account-wide GitHub OIDC provider."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Extra tags."
  type        = map(string)
  default     = {}
}

variable "bundle_run_id" {
  description = "Model bundle the API serves (immutable run id under bundles/). CD passes the last promoted one; empty serves no model (/ready 503)."
  type        = string
  default     = ""
}
