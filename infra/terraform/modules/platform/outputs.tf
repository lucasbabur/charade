output "api_endpoint" {
  description = "Internal HTTPS endpoint of the ranking API."
  value       = "https://${module.api.alb_dns_name}"
}

output "artifacts_bucket" {
  description = "Artifacts bucket (runs/ and bundles/)."
  value       = module.artifacts_bucket.bucket_name
}

output "ecr_repositories" {
  description = "Image repositories."
  value       = module.ecr.repository_urls
}

output "deploy_role_arn" {
  description = "Role GitHub Actions assumes."
  value       = module.ci.deploy_role_arn
}

output "alarm_topic_arn" {
  description = "Subscribe on-call here."
  value       = module.observability.alarm_topic_arn
}
