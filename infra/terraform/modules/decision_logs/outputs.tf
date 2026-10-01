output "stream_arn" {
  description = "Firehose delivery stream ARN."
  value       = aws_kinesis_firehose_delivery_stream.decisions.arn
}
