# Serving logs (the training and off-policy-evaluation data of tomorrow): the API writes JSON
# "decision", "impression" and "click" events to stdout; a subscription filter streams them through
# Firehose into S3, partitioned by hour, compressed. charade.data.events joins them into training rows.

data "aws_iam_policy_document" "assume_firehose" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["firehose.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "firehose" {
  name               = "${var.name}-decisions-firehose"
  assume_role_policy = data.aws_iam_policy_document.assume_firehose.json
  tags               = var.tags
}

data "aws_iam_policy_document" "firehose" {
  statement {
    actions   = ["s3:AbortMultipartUpload", "s3:GetBucketLocation", "s3:ListBucket", "s3:PutObject"]
    resources = [var.bucket_arn, "${var.bucket_arn}/decisions/*", "${var.bucket_arn}/decisions-errors/*"]
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [var.kms_key_arn]
  }
}

resource "aws_iam_role_policy" "firehose" {
  name   = "write-decisions"
  role   = aws_iam_role.firehose.id
  policy = data.aws_iam_policy_document.firehose.json
}

resource "aws_kinesis_firehose_delivery_stream" "decisions" {
  name        = "${var.name}-decisions"
  destination = "extended_s3"
  tags        = var.tags

  server_side_encryption {
    enabled  = true
    key_type = "CUSTOMER_MANAGED_CMK"
    key_arn  = var.kms_key_arn
  }

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose.arn
    bucket_arn          = var.bucket_arn
    prefix              = "decisions/dt=!{timestamp:yyyy-MM-dd}/hour=!{timestamp:HH}/"
    error_output_prefix = "decisions-errors/!{firehose:error-output-type}/dt=!{timestamp:yyyy-MM-dd}/"
    buffering_size      = 64
    buffering_interval  = 300
    compression_format  = "GZIP"
    kms_key_arn         = var.kms_key_arn

    # CloudWatch Logs hands Firehose gzipped subscription envelopes. Decompress them, keep only each log
    # event's message, and end every record with a newline, so S3 holds gzipped JSON lines: the exact
    # input of charade.data.events.
    processing_configuration {
      enabled = true
      processors {
        type = "Decompression"
        parameters {
          parameter_name  = "CompressionFormat"
          parameter_value = "GZIP"
        }
      }
      processors {
        type = "CloudWatchLogProcessing"
        parameters {
          parameter_name  = "DataMessageExtraction"
          parameter_value = "true"
        }
      }
      processors {
        type = "AppendDelimiterToRecord"
      }
    }
  }
}

data "aws_iam_policy_document" "assume_logs" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["logs.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "logs" {
  name               = "${var.name}-decisions-logs"
  assume_role_policy = data.aws_iam_policy_document.assume_logs.json
  tags               = var.tags
}

data "aws_iam_policy_document" "logs" {
  statement {
    actions   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
    resources = [aws_kinesis_firehose_delivery_stream.decisions.arn]
  }
}

resource "aws_iam_role_policy" "logs" {
  name   = "to-firehose"
  role   = aws_iam_role.logs.id
  policy = data.aws_iam_policy_document.logs.json
}

resource "aws_cloudwatch_log_subscription_filter" "decisions" {
  name            = "${var.name}-decisions"
  log_group_name  = var.api_log_group_name
  filter_pattern  = "{ ($.event = \"decision\") || ($.event = \"impression\") || ($.event = \"click\") }"
  destination_arn = aws_kinesis_firehose_delivery_stream.decisions.arn
  role_arn        = aws_iam_role.logs.arn
}
