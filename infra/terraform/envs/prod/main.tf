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

  environment                 = "prod"
  vpc_cidr                    = var.vpc_cidr
  single_nat_gateway          = false
  caller_cidrs                = var.caller_cidrs
  certificate_arn             = var.certificate_arn
  api_image_tag               = var.api_image_tag
  train_image_tag             = var.train_image_tag
  api_min_tasks               = 3
  api_max_tasks               = 30
  redis_node_type             = "cache.r7g.large"
  redis_replicas              = 2
  deletion_protection         = true
  create_github_oidc_provider = false
}
