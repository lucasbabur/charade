variable "name" {
  description = "Name prefix."
  type        = string
}

variable "bucket_arn" {
  description = "Destination bucket ARN."
  type        = string
}

variable "api_log_group_name" {
  description = "Log group the API writes decision events to."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key for the stream and objects."
  type        = string
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
