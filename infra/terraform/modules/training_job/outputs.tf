output "task_definition_arn" {
  description = "Training task definition."
  value       = aws_ecs_task_definition.train.arn
}

output "log_group_name" {
  description = "Training log group."
  value       = aws_cloudwatch_log_group.train.name
}
