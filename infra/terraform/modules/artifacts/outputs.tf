output "bucket_arn" {
  description = "ARN of the artifacts bucket."
  value       = aws_s3_bucket.this.arn
}

output "bucket_name" {
  description = "Name of the artifacts bucket."
  value       = aws_s3_bucket.this.bucket
}
