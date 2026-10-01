output "api_endpoint" {
  description = "Internal HTTPS endpoint."
  value       = module.charade.api_endpoint
}

output "deploy_role_arn" {
  description = "Role for GitHub Actions."
  value       = module.charade.deploy_role_arn
}

output "alarm_topic_arn" {
  description = "Alarm topic."
  value       = module.charade.alarm_topic_arn
}
