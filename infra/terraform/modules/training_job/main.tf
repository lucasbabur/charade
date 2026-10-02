# Daily retraining (docs/07: a frozen model loses ~0.003 NE per day). EventBridge Scheduler starts a
# Fargate task from the training image that runs: train -> ope -> parity -> drift ->
# mlcheck -> promote (scripts/retrain.sh, scripts/promote.sh). Only if every blocking gate passes, it
# uploads an immutable bundles/<run_id>/, registers an API task definition revision pinned to that run
# and deploys it; a failed gate or rollout leaves production on the previous revision and bundle.

data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "train" {
  name              = "/ecs/${var.name}-train"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
  tags              = var.tags
}

data "aws_iam_policy_document" "assume_ecs" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-train-execution"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-train-task"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = var.tags
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ReadTrainingData"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [var.data_bucket_arn, "${var.data_bucket_arn}/*"]
  }
  statement {
    sid       = "WriteRunsAndPromote"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/runs/*", "${var.artifacts_bucket_arn}/bundles/*"]
  }
  statement {
    sid       = "RedeployApi"
    actions   = ["ecs:UpdateService", "ecs:DescribeServices"]
    resources = [var.api_service_arn]
  }
  statement {
    sid       = "PinBundleRevision"
    actions   = ["ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition"]
    resources = ["*"] # neither action supports resource-level permissions
  }
  statement {
    sid       = "PassApiRoles"
    actions   = ["iam:PassRole"]
    resources = var.api_role_arns
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [var.kms_key_arn]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "train"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

resource "aws_ecs_task_definition" "train" {
  #checkov:skip=CKV_AWS_336:Training needs a writable workspace (data, artifacts) on ephemeral storage; it serves no traffic
  family                   = "${var.name}-train"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  tags                     = var.tags

  ephemeral_storage {
    size_in_gib = 50
  }

  container_definitions = jsonencode([{
    name      = "train"
    image     = var.image
    essential = true
    command   = ["./scripts/retrain.sh"]
    environment = [
      { name = "CHARADE_DATA_URI", value = var.data_uri },
      { name = "CHARADE_ARTIFACTS_URI", value = "s3://${var.artifacts_bucket_name}" },
      { name = "CHARADE_ECS_CLUSTER", value = var.cluster_arn },
      { name = "CHARADE_ECS_SERVICE", value = var.api_service_arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.train.name
        awslogs-region        = data.aws_region.current.region
        awslogs-stream-prefix = "train"
      }
    }
  }])
}

data "aws_iam_policy_document" "assume_scheduler" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.name}-train-scheduler"
  assume_role_policy = data.aws_iam_policy_document.assume_scheduler.json
  tags               = var.tags
}

data "aws_iam_policy_document" "scheduler" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.train.arn_without_revision, "${aws_ecs_task_definition.train.arn_without_revision}:*"]
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
  }
}

resource "aws_iam_role_policy" "scheduler" {
  name   = "run-training"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler.json
}

resource "aws_security_group" "train" {
  #checkov:skip=CKV2_AWS_5:Attached through the scheduler target's network configuration, which checkov does not follow
  name        = "${var.name}-train"
  description = "Training task: egress to AWS endpoints only"
  vpc_id      = var.vpc_id
  tags        = var.tags

  egress {
    description = "HTTPS to VPC endpoints"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_scheduler_schedule" "daily" {
  name                         = "${var.name}-daily-retrain"
  kms_key_arn                  = var.kms_key_arn
  schedule_expression          = var.schedule
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = var.cluster_arn
    role_arn = aws_iam_role.scheduler.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.train.arn
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = var.private_subnet_ids
        security_groups  = [aws_security_group.train.id]
        assign_public_ip = false
      }
    }

    retry_policy {
      maximum_retry_attempts = 1
    }
  }
}
