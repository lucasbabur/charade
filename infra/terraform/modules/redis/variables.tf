variable "name" {
  description = "Name prefix."
  type        = string
}

variable "vpc_id" {
  description = "VPC id."
  type        = string
}

variable "subnet_ids" {
  description = "Private subnets for the cache nodes."
  type        = list(string)
}

variable "client_security_group_ids" {
  description = "Security groups allowed to connect (the API tasks)."
  type        = list(string)
}

variable "node_type" {
  description = "Cache node type."
  type        = string
  default     = "cache.t4g.medium"
}

variable "replicas" {
  description = "Read replicas per shard."
  type        = number
  default     = 1
}

variable "kms_key_arn" {
  description = "KMS key for encryption at rest and the AUTH secret."
  type        = string
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
