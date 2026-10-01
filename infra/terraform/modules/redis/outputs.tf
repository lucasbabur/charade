output "url_secret_arn" {
  description = "Secrets Manager ARN of the full rediss:// URL (injected as CHARADE_REDIS_URL)."
  value       = aws_secretsmanager_secret.url.arn
}

output "security_group_id" {
  description = "Redis security group."
  value       = aws_security_group.this.id
}
