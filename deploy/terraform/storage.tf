# S3: the labelled holdout the producer replays, and the audit archive of every decision (FR-API-03, NFR-11).
#   data   s3://<name>-data-<account>/holdout/*.parquet    copied once by `ops.py dataset`
#   audit  s3://<name>-audit-<account>/decisions/...      Kinesis Firehose from <name>-decisions, gzip JSON lines,
#                                                         Glacier after 90 days, kept audit_retention_days
# Production hardening (not here, because it blocks `ops.py down`): S3 Object Lock in compliance mode on the audit bucket.

data "aws_caller_identity" "me" {}

locals {
  buckets = { data = "${local.name}-data-${data.aws_caller_identity.me.account_id}",
  audit = "${local.name}-audit-${data.aws_caller_identity.me.account_id}" }
}

resource "aws_s3_bucket" "this" {
  for_each      = local.buckets
  bucket        = each.value
  force_destroy = true # `ops.py down` removes this project completely
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_versioning" "audit" {
  bucket = aws_s3_bucket.this["audit"].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "audit" {
  bucket = aws_s3_bucket.this["audit"].id
  rule {
    id     = "audit-retention"
    status = "Enabled"
    filter {}
    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
    expiration { days = var.audit_retention_days }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}

# ---------------------------------------------------------------- Firehose: decisions stream -> audit bucket
data "aws_iam_policy_document" "firehose_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["firehose.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "firehose" {
  name               = "${local.name}-firehose"
  assume_role_policy = data.aws_iam_policy_document.firehose_assume.json
}

resource "aws_iam_role_policy" "firehose" {
  name = "${local.name}-firehose"
  role = aws_iam_role.firehose.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Resource = aws_kinesis_stream.this["decisions"].arn,
    Action = ["kinesis:DescribeStream", "kinesis:GetShardIterator", "kinesis:GetRecords", "kinesis:ListShards"] },
    { Effect = "Allow", Action = ["kms:Decrypt"], Resource = "*",
    Condition = { StringEquals = { "kms:ViaService" = "kinesis.${var.region}.amazonaws.com" } } },
    { Effect = "Allow", Action = ["s3:AbortMultipartUpload", "s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket",
      "s3:ListBucketMultipartUploads", "s3:PutObject"],
    Resource = [aws_s3_bucket.this["audit"].arn, "${aws_s3_bucket.this["audit"].arn}/*"] },
    { Effect = "Allow", Action = ["logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.firehose.arn}:*" },
  ] })
}

resource "aws_cloudwatch_log_group" "firehose" {
  name              = "/aws/kinesisfirehose/${local.name}-audit"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_stream" "firehose" {
  name           = "DestinationDelivery"
  log_group_name = aws_cloudwatch_log_group.firehose.name
}

resource "aws_kinesis_firehose_delivery_stream" "audit" {
  name        = "${local.name}-audit"
  destination = "extended_s3"
  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.this["decisions"].arn
    role_arn           = aws_iam_role.firehose.arn
  }
  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose.arn
    bucket_arn          = aws_s3_bucket.this["audit"].arn
    prefix              = "decisions/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"
    error_output_prefix = "errors/!{firehose:error-output-type}/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"
    compression_format  = "GZIP"
    buffering_interval  = 60
    buffering_size      = 5
    cloudwatch_logging_options {
      enabled         = true
      log_group_name  = aws_cloudwatch_log_group.firehose.name
      log_stream_name = aws_cloudwatch_log_stream.firehose.name
    }
  }
  depends_on = [aws_iam_role_policy.firehose]
}
