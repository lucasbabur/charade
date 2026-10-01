# Private, versioned, encrypted bucket. KMS by default; SSE-S3 when `kms_key_arn` is null (ALB
# access logs only support SSE-S3). Optional lifecycle expiry and ALB log-delivery permission.

data "aws_elb_service_account" "this" {}

resource "aws_s3_bucket" "this" {
  #checkov:skip=CKV2_AWS_62:Event notifications are not needed; consumers poll on schedule
  #checkov:skip=CKV_AWS_144:Cross-region replication is a disaster-recovery decision outside this sketch
  bucket = var.bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "this" {
  bucket = aws_s3_bucket.this.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_arn == null ? "AES256" : "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = var.kms_key_arn != null
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  bucket                  = aws_s3_bucket.this.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_logging" "this" {
  count         = var.access_log_bucket == null ? 0 : 1
  bucket        = aws_s3_bucket.this.id
  target_bucket = var.access_log_bucket
  target_prefix = "s3/${var.bucket_name}/"
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  rule {
    id     = "expire-noncurrent-and-incomplete"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  dynamic "rule" {
    for_each = var.expire_days == null ? [] : [var.expire_days]
    content {
      id     = "expire-current"
      status = "Enabled"
      filter {}
      expiration {
        days = rule.value
      }
      abort_incomplete_multipart_upload {
        days_after_initiation = 7
      }
    }
  }
}

data "aws_iam_policy_document" "this" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.this.arn, "${aws_s3_bucket.this.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  dynamic "statement" {
    for_each = var.alb_log_delivery ? [1] : []
    content {
      sid       = "AlbLogDelivery"
      actions   = ["s3:PutObject"]
      resources = ["${aws_s3_bucket.this.arn}/alb/*"]
      principals {
        type        = "AWS"
        identifiers = [data.aws_elb_service_account.this.arn]
      }
    }
  }
}

resource "aws_s3_bucket_policy" "this" {
  bucket = aws_s3_bucket.this.id
  policy = data.aws_iam_policy_document.this.json
}
