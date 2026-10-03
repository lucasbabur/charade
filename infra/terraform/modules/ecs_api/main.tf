# The ranking API: Fargate service behind an ALB, health-checked on /ready, autoscaled on CPU and
# request count, deployed with a circuit breaker that rolls back failed releases.

data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.name}-api"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
  tags              = var.tags
}

resource "aws_ecs_cluster" "this" {
  name = var.name
  tags = var.tags

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Internal ALB: HTTPS from the ad-serving VPC ranges"
  vpc_id      = var.vpc_id
  tags        = var.tags

  ingress {
    description = "HTTPS from callers"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.caller_cidrs
  }

  egress {
    description = "To API tasks"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_security_group" "task" {
  name        = "${var.name}-api"
  description = "API tasks: inbound from the ALB only"
  vpc_id      = var.vpc_id
  tags        = var.tags

  ingress {
    description     = "HTTP from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "HTTPS to AWS endpoints, Redis TLS inside the VPC"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_lb" "this" {
  #checkov:skip=CKV2_AWS_28:Internal ALB inside the ad-serving VPC; WAF belongs at the public edge
  name                       = "${var.name}-api"
  internal                   = true
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = var.private_subnet_ids
  drop_invalid_header_fields = true
  enable_deletion_protection = var.deletion_protection
  tags                       = var.tags

  access_logs {
    bucket  = var.access_logs_bucket
    prefix  = "alb"
    enabled = true
  }
}

resource "aws_lb_target_group" "api" {
  name                 = "${var.name}-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = var.vpc_id
  deregistration_delay = 15
  tags                 = var.tags

  health_check {
    path                = "/ready"
    matcher             = "200"
    interval            = 10
    timeout             = 3
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
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
  name               = "${var.name}-api-execution"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.redis_url_secret_arn]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [var.kms_key_arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-api-task"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = var.tags
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ReadProductionBundle"
    actions   = ["s3:GetObject"]
    resources = ["${var.artifacts_bucket_arn}/bundles/*"]
  }
  statement {
    sid       = "ListBundles"
    actions   = ["s3:ListBucket"]
    resources = [var.artifacts_bucket_arn]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [var.kms_key_arn]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "bundle-read"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  tags                     = var.tags

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  # An init container copies the bundle from S3 into a task-local volume; the API container starts only
  # after it succeeds and mounts the bundle read-only. No S3 code in the application. The bundle's run id
  # is part of the task definition, so a revision names exactly one model: the circuit breaker's rollback
  # to the previous revision also restores the previous model (scripts/promote.sh registers revisions).
  container_definitions = jsonencode([
    {
      name                   = "fetch-bundle"
      image                  = var.bundle_fetch_image
      essential              = false
      readonlyRootFilesystem = true
      entryPoint             = ["sh", "-c"]
      command                = ["[ -z \"$BUNDLE_RUN_ID\" ] || aws s3 cp --recursive --only-show-errors ${var.bundles_uri}$BUNDLE_RUN_ID/ /bundle/current/"]
      environment            = [{ name = "BUNDLE_RUN_ID", value = var.bundle_run_id }]
      mountPoints            = [{ sourceVolume = "bundle", containerPath = "/bundle", readOnly = false }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "fetch"
        }
      }
    },
    {
      name                   = "api"
      image                  = var.image
      essential              = true
      readonlyRootFilesystem = true
      dependsOn              = [{ containerName = "fetch-bundle", condition = "SUCCESS" }]
      portMappings           = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "WEB_CONCURRENCY", value = tostring(var.workers) },
        { name = "CHARADE_ARTIFACTS_DIR", value = "/app/artifacts/current" },
      ]
      secrets     = [{ name = "CHARADE_REDIS_URL", valueFrom = var.redis_url_secret_arn }]
      mountPoints = [{ sourceVolume = "bundle", containerPath = "/app/artifacts", readOnly = true }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "api"
        }
      }
    },
  ])

  volume {
    name = "bundle"
  }
}

resource "aws_ecs_service" "api" {
  name                              = "${var.name}-api"
  cluster                           = aws_ecs_cluster.this.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = var.min_tasks
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 30
  propagate_tags                    = "SERVICE"
  tags                              = var.tags

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [desired_count]
  }
}

resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.min_tasks
  max_capacity       = var.max_tasks
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.name}-api-cpu"
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    target_value = 50
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_policy" "requests" {
  name               = "${var.name}-api-requests"
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    # Measured: p99 25-31 ms at 400 rps per 8-worker task (docs/07-serving-operations.md).
    target_value = var.target_rps_per_task * 60
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.api.arn_suffix}"
    }
  }
}
