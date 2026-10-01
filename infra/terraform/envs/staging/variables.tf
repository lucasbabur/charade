variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "account_suffix" {
  description = "Suffix that makes bucket names globally unique (e.g. the AWS account id)."
  type        = string
}
