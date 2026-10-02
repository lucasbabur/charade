# Online user-history store: Redis 7 replication group, multi-AZ, encrypted at rest and in transit,
# AUTH token in Secrets Manager, reachable only from the API security group.

resource "random_password" "auth" {
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "url" {
  #checkov:skip=CKV2_AWS_57:Rotation requires a coordinated ElastiCache AUTH update; rotated by runbook (docs/07-serving-operations.md)
  name       = "${var.name}/redis-url"
  kms_key_id = var.kms_key_arn
  tags       = var.tags
}

# The API reads one setting, CHARADE_REDIS_URL, so the secret holds the full TLS URL with the token.
resource "aws_secretsmanager_secret_version" "url" {
  secret_id     = aws_secretsmanager_secret.url.id
  secret_string = "rediss://:${random_password.auth.result}@${aws_elasticache_replication_group.this.primary_endpoint_address}:6379/0"
}

resource "aws_elasticache_subnet_group" "this" {
  name       = var.name
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

resource "aws_security_group" "this" {
  name        = "${var.name}-redis"
  description = "Redis: inbound from the API only"
  vpc_id      = var.vpc_id
  tags        = var.tags

  ingress {
    description     = "Redis from API tasks"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = var.client_security_group_ids
  }
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id       = var.name
  description                = "Charade user-history counters"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = var.node_type
  num_cache_clusters         = var.replicas + 1
  automatic_failover_enabled = true
  multi_az_enabled           = true
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.this.name
  security_group_ids         = [aws_security_group.this.id]
  at_rest_encryption_enabled = true
  kms_key_id                 = var.kms_key_arn
  transit_encryption_enabled = true
  auth_token                 = random_password.auth.result
  snapshot_retention_limit   = 3
  auto_minor_version_upgrade = true
  tags                       = var.tags
}
