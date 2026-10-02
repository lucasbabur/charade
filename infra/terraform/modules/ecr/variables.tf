variable "name" {
  description = "Name prefix."
  type        = string
}

variable "repositories" {
  description = "Repository suffixes."
  type        = list(string)
  default     = ["api"]
}

variable "keep_images" {
  description = "Images retained per repository."
  type        = number
  default     = 20
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
