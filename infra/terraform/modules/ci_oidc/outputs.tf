output "deploy_role_arn" {
  description = "Role for aws-actions/configure-aws-credentials."
  value       = aws_iam_role.deploy.arn
}
