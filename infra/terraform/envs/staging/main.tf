terraform {
  required_version = ">= 1.13"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  # Partial config: bucket, key and region are passed at `terraform init -backend-config=...`.
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = local.tags
  }
}

locals {
  tags = {
    project     = "charade"
    environment = "staging"
    managed_by  = "terraform"
  }
}

module "artifacts" {
  source      = "../../modules/artifacts"
  bucket_name = "charade-staging-artifacts-${var.account_suffix}"
  tags        = local.tags
}
