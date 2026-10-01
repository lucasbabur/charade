variable "name" {
  description = "Name prefix."
  type        = string
}

variable "region" {
  description = "AWS region (dashboard widgets)."
  type        = string
}

variable "alb_arn_suffix" {
  description = "ALB ARN suffix."
  type        = string
}

variable "target_group_arn_suffix" {
  description = "Target group ARN suffix."
  type        = string
}

variable "api_log_group_name" {
  description = "API log group (decision events)."
  type        = string
}

variable "train_log_group_name" {
  description = "Training log group (mlcheck output)."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key for the SNS topic."
  type        = string
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
