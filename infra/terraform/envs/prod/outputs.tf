output "api_endpoint" {
  description = "Internal HTTPS endpoint."
  value       = module.charade.api_endpoint
}

output "alarm_topic_arn" {
  description = "Alarm topic."
  value       = module.charade.alarm_topic_arn
}

output "artifacts_bucket" {
  description = "Artifacts bucket; CD reads bundles/CURRENT from it to keep the served bundle on apply."
  value       = module.charade.artifacts_bucket
}
