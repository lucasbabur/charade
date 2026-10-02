variable "name" {
  description = "Name prefix."
  type        = string
}

variable "cluster_arn" {
  description = "ECS cluster that runs the task."
  type        = string
}

variable "api_role_arns" {
  description = "API execution and task roles; promotion registers task definition revisions that use them."
  type        = list(string)
}

variable "api_service_arn" {
  description = "API service the job redeploys after promoting a bundle."
  type        = string
}

variable "vpc_id" {
  description = "VPC id."
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets."
  type        = list(string)
}

variable "image" {
  description = "Training image (ECR URI, immutable tag)."
  type        = string
}

variable "data_uri" {
  description = "s3:// prefix of the impression and character exports."
  type        = string
}

variable "data_bucket_arn" {
  description = "Bucket holding the exports."
  type        = string
}

variable "artifacts_bucket_arn" {
  description = "Artifacts bucket ARN."
  type        = string
}

variable "artifacts_bucket_name" {
  description = "Artifacts bucket name."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key."
  type        = string
}

variable "schedule" {
  description = "When to retrain (after the day's logs are complete)."
  type        = string
  default     = "cron(30 2 * * ? *)"
}

variable "cpu" {
  description = "Task vCPU units. Fargate has no GPU; the DCN trains on 16 vCPU in minutes at this scale."
  type        = number
  default     = 16384
}

variable "memory" {
  description = "Task memory (MiB)."
  type        = number
  default     = 65536
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
