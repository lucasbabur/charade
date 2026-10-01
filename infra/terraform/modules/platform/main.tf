# One Charade environment: network, encryption, storage, registry, Redis, API, decision logs,
# daily retraining, alarms and CI access. Environments differ only in sizing variables.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "zone-id"
    values = var.zone_ids
  }
}

locals {
  azs  = slice(data.aws_availability_zones.available.names, 0, 3)
  tags = merge(var.tags, { project = "charade", environment = var.environment, managed_by = "terraform" })
  name = "charade-${var.environment}"
}

module "vpc" {
  #checkov:skip=CKV_TF_1:Pinned by registry version constraint; Dependabot proposes upgrades with review
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.7"

  name                 = local.name
  cidr                 = var.vpc_cidr
  azs                  = local.azs
  private_subnets      = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i)]
  public_subnets       = [for i in range(3) : cidrsubnet(var.vpc_cidr, 8, 48 + i)]
  enable_nat_gateway   = true
  single_nat_gateway   = var.single_nat_gateway
  enable_dns_hostnames = true
  tags                 = local.tags

  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60
}

# Key policies scope to the key itself: "*" as a resource means "this key" (AWS key-policy semantics).
data "aws_iam_policy_document" "kms" {
  #checkov:skip=CKV_AWS_356:Key policy resource "*" refers to this key only
  #checkov:skip=CKV_AWS_109:Account root administers its own key, the AWS default key policy
  #checkov:skip=CKV_AWS_111:Same as above
  statement {
    sid       = "AccountAdministration"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
  }
  statement {
    sid       = "CloudWatchLogs"
    actions   = ["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["logs.${data.aws_region.current.region}.amazonaws.com"]
    }
  }
}

resource "aws_kms_key" "this" {
  description             = "${local.name} data, logs and secrets"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy                  = data.aws_iam_policy_document.kms.json
  tags                    = local.tags
}

resource "aws_kms_alias" "this" {
  name          = "alias/${local.name}"
  target_key_id = aws_kms_key.this.key_id
}

module "logs_bucket" {
  #checkov:skip=CKV_AWS_145:ALB access logs require SSE-S3
  source           = "../bucket"
  bucket_name      = "${local.name}-logs-${data.aws_caller_identity.current.account_id}"
  alb_log_delivery = true
  expire_days      = 90
  tags             = local.tags
}

module "artifacts_bucket" {
  source            = "../bucket"
  bucket_name       = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
  kms_key_arn       = aws_kms_key.this.arn
  access_log_bucket = module.logs_bucket.bucket_name
  tags              = local.tags
}

module "data_bucket" {
  source            = "../bucket"
  bucket_name       = "${local.name}-data-${data.aws_caller_identity.current.account_id}"
  kms_key_arn       = aws_kms_key.this.arn
  access_log_bucket = module.logs_bucket.bucket_name
  expire_days       = var.data_retention_days
  tags              = local.tags
}

module "ecr" {
  source = "../ecr"
  name   = local.name
  tags   = local.tags
}

module "api" {
  source               = "../ecs_api"
  name                 = local.name
  vpc_id               = module.vpc.vpc_id
  vpc_cidr             = var.vpc_cidr
  private_subnet_ids   = module.vpc.private_subnets
  caller_cidrs         = var.caller_cidrs
  certificate_arn      = var.certificate_arn
  image                = "${module.ecr.repository_urls["api"]}:${var.api_image_tag}"
  bundle_uri           = "s3://${module.artifacts_bucket.bucket_name}/bundles/current/"
  artifacts_bucket_arn = module.artifacts_bucket.bucket_arn
  access_logs_bucket   = module.logs_bucket.bucket_name
  redis_url_secret_arn = module.redis.url_secret_arn
  kms_key_arn          = aws_kms_key.this.arn
  min_tasks            = var.api_min_tasks
  max_tasks            = var.api_max_tasks
  deletion_protection  = var.deletion_protection
  tags                 = local.tags
}

module "redis" {
  source                    = "../redis"
  name                      = local.name
  vpc_id                    = module.vpc.vpc_id
  subnet_ids                = module.vpc.private_subnets
  client_security_group_ids = [module.api.task_security_group_id]
  node_type                 = var.redis_node_type
  replicas                  = var.redis_replicas
  kms_key_arn               = aws_kms_key.this.arn
  tags                      = local.tags
}

module "decision_logs" {
  source             = "../decision_logs"
  name               = local.name
  bucket_arn         = module.data_bucket.bucket_arn
  api_log_group_name = module.api.log_group_name
  kms_key_arn        = aws_kms_key.this.arn
  tags               = local.tags
}

module "training" {
  source                = "../training_job"
  name                  = local.name
  cluster_arn           = module.api.cluster_arn
  vpc_id                = module.vpc.vpc_id
  vpc_cidr              = var.vpc_cidr
  private_subnet_ids    = module.vpc.private_subnets
  image                 = "${module.ecr.repository_urls["train"]}:${var.train_image_tag}"
  data_uri              = "s3://${module.data_bucket.bucket_name}/exports/"
  data_bucket_arn       = module.data_bucket.bucket_arn
  artifacts_bucket_arn  = module.artifacts_bucket.bucket_arn
  artifacts_bucket_name = module.artifacts_bucket.bucket_name
  kms_key_arn           = aws_kms_key.this.arn
  tags                  = local.tags
}

module "observability" {
  source                  = "../observability"
  name                    = local.name
  region                  = data.aws_region.current.region
  alb_arn_suffix          = module.api.alb_arn_suffix
  target_group_arn_suffix = module.api.target_group_arn_suffix
  api_log_group_name      = module.api.log_group_name
  train_log_group_name    = module.training.log_group_name
  kms_key_arn             = aws_kms_key.this.arn
  tags                    = local.tags
}

module "ci" {
  source              = "../ci_oidc"
  name                = local.name
  repository          = var.github_repository
  create_provider     = var.create_github_oidc_provider
  ecr_repository_arns = values(module.ecr.repository_arns)
  task_role_arns      = module.api.task_role_arns
  tags                = local.tags
}
