terraform {
  required_version = ">= 1.13"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }
  # Partial configuration: `terraform init -backend-config=backend.hcl` supplies bucket, key, region.
  backend "s3" {
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.region
}

module "charade" {
  source = "../../modules/platform"

  environment         = "staging"
  vpc_cidr            = var.vpc_cidr
  single_nat_gateway  = true
  caller_cidrs        = var.caller_cidrs
  certificate_arn     = var.certificate_arn
  api_image_tag       = var.api_image_tag
  bundle_run_id       = var.bundle_run_id
  api_min_tasks       = 2
  api_max_tasks       = 4
  redis_node_type     = "cache.t4g.medium"
  redis_replicas      = 1
  deletion_protection = false
}
