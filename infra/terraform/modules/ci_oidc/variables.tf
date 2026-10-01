variable "name" {
  description = "Name prefix."
  type        = string
}

variable "repository" {
  description = "GitHub repository (owner/name) allowed to assume the role."
  type        = string
}

variable "create_provider" {
  description = "Create the account-wide GitHub OIDC provider (only one per account)."
  type        = bool
  default     = false
}

variable "ecr_repository_arns" {
  description = "Repositories CI may push to."
  type        = list(string)
}

variable "task_role_arns" {
  description = "Task roles CI may pass when registering task definitions."
  type        = list(string)
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
