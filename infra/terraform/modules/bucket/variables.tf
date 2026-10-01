variable "bucket_name" {
  description = "Globally unique S3 bucket name."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key for SSE-KMS; null selects SSE-S3."
  type        = string
  default     = null
}

variable "expire_days" {
  description = "Expire current objects after this many days (null keeps them)."
  type        = number
  default     = null
}

variable "alb_log_delivery" {
  description = "Allow the regional ELB account to write access logs under alb/."
  type        = bool
  default     = false
}

variable "access_log_bucket" {
  description = "Bucket receiving S3 server access logs (null disables)."
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
