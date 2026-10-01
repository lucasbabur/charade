output "alarm_topic_arn" {
  description = "SNS topic that receives every alarm (subscribe the on-call tool)."
  value       = aws_sns_topic.alarms.arn
}
