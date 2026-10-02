variable "name" {
  description = "Name prefix."
  type        = string
}

variable "vpc_id" {
  description = "VPC id."
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR (task egress stays inside it via VPC endpoints)."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets for the ALB and tasks."
  type        = list(string)
}

variable "caller_cidrs" {
  description = "CIDRs of the ad servers allowed to call the API."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for the internal HTTPS listener."
  type        = string
}

variable "image" {
  description = "API image (ECR URI with an immutable tag)."
  type        = string
}

variable "bundles_uri" {
  description = "s3:// prefix holding immutable bundles/<run_id>/ and the CURRENT record (trailing slash)."
  type        = string
}

variable "bundle_run_id" {
  description = "Bundle pinned into the task definition; empty starts tasks without a model (not ready)."
  type        = string
  default     = ""
}

variable "bundle_fetch_image" {
  description = "Image with the AWS CLI used by the init container (pin by digest in production)."
  type        = string
  default     = "public.ecr.aws/aws-cli/aws-cli:2.27.50"
}

variable "artifacts_bucket_arn" {
  description = "Artifacts bucket ARN (bundles are read from it)."
  type        = string
}

variable "access_logs_bucket" {
  description = "Bucket for ALB access logs."
  type        = string
}

variable "redis_url_secret_arn" {
  description = "Secret holding CHARADE_REDIS_URL."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key for logs and secrets."
  type        = string
}

variable "cpu" {
  description = "Task vCPU units (8192 = 8 vCPU, matching the load-tested 8 workers)."
  type        = number
  default     = 8192
}

variable "memory" {
  description = "Task memory (MiB)."
  type        = number
  default     = 16384
}

variable "workers" {
  description = "Uvicorn workers per task (one per vCPU)."
  type        = number
  default     = 8
}

variable "min_tasks" {
  description = "Minimum tasks (spread across AZs)."
  type        = number
  default     = 3
}

variable "max_tasks" {
  description = "Maximum tasks."
  type        = number
  default     = 30
}

variable "target_rps_per_task" {
  description = "Requests per second per task the request-count policy tracks."
  type        = number
  default     = 400
}

variable "deletion_protection" {
  description = "ALB deletion protection."
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 365
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
