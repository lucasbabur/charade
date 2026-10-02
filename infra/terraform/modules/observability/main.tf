# Alarms that page (docs/07-serving-operations.md): latency SLO, errors, degraded serving and no-fill, plus a
# dashboard. Application metrics come from the structured decision log, so no
# metrics agent is needed for the alarms; /metrics stays available for Prometheus scraping.

resource "aws_sns_topic" "alarms" {
  name              = "${var.name}-alarms"
  kms_master_key_id = var.kms_key_arn
  tags              = var.tags
}

resource "aws_cloudwatch_metric_alarm" "latency_p99" {
  alarm_name          = "${var.name}-api-p99-latency"
  alarm_description   = "ALB p99 above 40 ms for 5 minutes (SLO: 50 ms). Runbook: docs/07-serving-operations.md#latency"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p99"
  dimensions          = { LoadBalancer = var.alb_arn_suffix, TargetGroup = var.target_group_arn_suffix }
  period              = 60
  evaluation_periods  = 5
  threshold           = 0.04
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${var.name}-api-5xx-rate"
  alarm_description   = "Target 5xx above 0.5 % of requests for 5 minutes. Runbook: docs/07-serving-operations.md#errors"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0.5
  evaluation_periods  = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags

  metric_query {
    id          = "rate"
    expression  = "100 * errors / MAX([errors, requests])"
    label       = "5xx %"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      dimensions  = { LoadBalancer = var.alb_arn_suffix, TargetGroup = var.target_group_arn_suffix }
      period      = 60
      stat        = "Sum"
    }
  }

  metric_query {
    id = "requests"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      dimensions  = { LoadBalancer = var.alb_arn_suffix, TargetGroup = var.target_group_arn_suffix }
      period      = 60
      stat        = "Sum"
    }
  }
}

locals {
  log_metrics = {
    degraded = { pattern = "{ $.event = \"decision\" && $.degraded IS TRUE }", threshold = 50, description = "Feature store unavailable: serving with cold-user defaults" }
    no_fill  = { pattern = "{ $.event = \"decision\" && $.chosen_id IS NULL }", threshold = 500, description = "Every candidate gated: check brand-safety matrix, caps, budgets" }
  }
}

resource "aws_cloudwatch_log_metric_filter" "decision" {
  for_each       = local.log_metrics
  name           = "${var.name}-${each.key}"
  log_group_name = var.api_log_group_name
  pattern        = each.value.pattern

  metric_transformation {
    name      = each.key
    namespace = "Charade"
    value     = "1"
    unit      = "Count"
  }
}

resource "aws_cloudwatch_metric_alarm" "decision" {
  for_each            = local.log_metrics
  alarm_name          = "${var.name}-${each.key}"
  alarm_description   = "${each.value.description}. Runbook: docs/07-serving-operations.md"
  namespace           = "Charade"
  metric_name         = each.key
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = each.value.threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags
  depends_on          = [aws_cloudwatch_log_metric_filter.decision]
}

resource "aws_cloudwatch_dashboard" "this" {
  dashboard_name = var.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", width = 12, height = 6
        properties = {
          title   = "Latency (ALB target)"
          region  = var.region
          metrics = [for p in ["p50", "p95", "p99"] : ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", var.alb_arn_suffix, "TargetGroup", var.target_group_arn_suffix, { stat = p }]]
        }
      },
      {
        type = "metric", width = 12, height = 6
        properties = {
          title   = "Degraded and no-fill decisions"
          region  = var.region
          metrics = [["Charade", "degraded", { stat = "Sum" }], ["Charade", "no_fill", { stat = "Sum" }]]
        }
      },
    ]
  })
}
