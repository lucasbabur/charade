output "cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.this.name
}

output "cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.this.arn
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.api.name
}

output "task_security_group_id" {
  description = "Security group of the API tasks."
  value       = aws_security_group.task.id
}

output "alb_arn_suffix" {
  description = "ALB ARN suffix (CloudWatch dimension)."
  value       = aws_lb.this.arn_suffix
}

output "target_group_arn_suffix" {
  description = "Target group ARN suffix (CloudWatch dimension)."
  value       = aws_lb_target_group.api.arn_suffix
}

output "log_group_name" {
  description = "API log group."
  value       = aws_cloudwatch_log_group.api.name
}

output "alb_dns_name" {
  description = "Internal ALB DNS name."
  value       = aws_lb.this.dns_name
}

output "task_role_arns" {
  description = "Execution and task role ARNs (CI needs iam:PassRole on them)."
  value       = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
}
