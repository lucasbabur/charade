variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR."
  type        = string
}

variable "caller_cidrs" {
  description = "CIDRs of the ad servers allowed to call the API."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate ARN for the internal ALB."
  type        = string
}

variable "api_image_tag" {
  description = "API image tag."
  type        = string
}

variable "bundle_run_id" {
  description = "Model bundle the API serves (immutable run id under bundles/). CD passes the last promoted one; empty serves no model (/ready 503)."
  type        = string
  default     = ""
}
