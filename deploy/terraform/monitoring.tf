# Logs, alerting (SNS) and a CloudWatch dashboard (NFR-13, MLOPS-02/04 for the online path).

resource "aws_cloudwatch_log_group" "stream" {
  name              = "/ecs/${local.name}/stream"
  retention_in_days = var.log_retention_days
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# The downstream hook: every BLOCK action is published here (subscribe a Lambda, a webhook, a queue, ...).
resource "aws_sns_topic" "actions" {
  name = "${local.name}-fraud-actions"
}

locals {
  alarm_actions = [aws_sns_topic.alerts.arn]
  stream_dims   = tomap({ Service = "stream" })
  alarms = {
    "scoring-unreachable" = { desc = "Transactions decided by the fail-open rule: the Fraud Model API did not answer (FR-API-01)",
    ns = local.cw_namespace, metric = "FailOpen", stat = "Sum", dims = local.stream_dims, period = 300, n = 1, threshold = 0 }
    "scoring-auth" = { desc = "The Fraud Model API rejected the stream's token (401): payments are held for REVIEW until ARGUS_SCORING_TOKEN matches",
    ns = local.cw_namespace, metric = "AuthErrors", stat = "Sum", dims = local.stream_dims, period = 300, n = 1, threshold = 0 }
    "block-spike" = { desc = "Unusually many BLOCK decisions (attack wave, or a bad threshold change)",
    ns = local.cw_namespace, metric = "Blocked", stat = "Sum", dims = local.stream_dims, period = 300, n = 1, threshold = 60 }
    "missed-fraud" = { desc = "Labelled fraud approved by the stream (ground truth from the producer)",
    ns = local.cw_namespace, metric = "FraudMissed", stat = "Sum", dims = local.stream_dims, period = 900, n = 1, threshold = 5 }
    "kinesis-lag" = { desc = "The scorer is more than 60 s behind the transactions stream",
      ns = "AWS/Kinesis", metric = "GetRecords.IteratorAgeMilliseconds", stat = "Maximum",
    dims = tomap({ StreamName = aws_kinesis_stream.this["transactions"].name }), period = 300, n = 2, threshold = 60000 }
    "task-cpu" = { desc = "Stream task CPU above 85%", ns = "AWS/ECS", metric = "CPUUtilization", stat = "Average",
    dims = tomap({ ClusterName = local.name, ServiceName = local.service_name }), period = 300, n = 3, threshold = 85 }
    "task-memory" = { desc = "Stream task memory above 85%", ns = "AWS/ECS", metric = "MemoryUtilization", stat = "Average",
    dims = tomap({ ClusterName = local.name, ServiceName = local.service_name }), period = 300, n = 3, threshold = 85 }
    "unhealthy" = { desc = "The stream service fails its load-balancer health check", ns = "AWS/ApplicationELB",
      metric = "UnHealthyHostCount", stat = "Maximum", period = 60, n = 5, threshold = 0,
    dims = tomap({ LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.stream.arn_suffix }) }
  }
}

resource "aws_cloudwatch_metric_alarm" "this" {
  for_each            = local.alarms
  alarm_name          = "${local.name}-${each.key}"
  alarm_description   = each.value.desc
  namespace           = each.value.ns
  metric_name         = each.value.metric
  dimensions          = each.value.dims
  statistic           = each.value.stat
  period              = each.value.period
  evaluation_periods  = each.value.n
  threshold           = each.value.threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

# OBJ-10: p99 of the Fraud Model API as the stream sees it (network included) above the 250 ms budget
resource "aws_cloudwatch_metric_alarm" "api_p99" {
  alarm_name          = "${local.name}-api-latency-p99"
  alarm_description   = "Fraud Model API p99 latency above 250 ms seen from the stream (OBJ-10)"
  namespace           = local.cw_namespace
  metric_name         = "ApiLatency"
  dimensions          = local.stream_dims
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2
  threshold           = 250
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_dashboard" "this" {
  dashboard_name = local.name
  dashboard_body = jsonencode({ widgets = [
    { type = "text", x = 0, y = 0, width = 24, height = 2, properties = {
    markdown = "# ARGUS ${var.environment}\nKinesis → stream scorer (Fargate) → Fraud Model API on the GPU VM (${local.scoring_url == "" ? "not set" : local.scoring_url}) → decision → action → decisions stream → S3 audit · API: http://${aws_lb.this.dns_name}/live/docs" } },
    { type = "metric", x = 0, y = 2, width = 8, height = 6, properties = {
      title = "Transactions and decisions / min", region = var.region, stat = "Sum", period = 60, view = "timeSeries",
    metrics = [for m in ["Transactions", "Allowed", "Reviewed", "Blocked", "FailOpen", "AuthErrors"] : [local.cw_namespace, m, "Service", "stream"]] } },
    { type = "metric", x = 8, y = 2, width = 8, height = 6, properties = {
      title   = "Fraud Model API latency (ms, as seen by the stream)", region = var.region, period = 60, view = "timeSeries",
      metrics = [for s in ["p50", "p95", "p99"] : [local.cw_namespace, "ApiLatency", "Service", "stream", { stat = s }]],
    annotations = { horizontal = [{ label = "OBJ-10 250 ms", value = 250 }] } } },
    { type = "metric", x = 16, y = 2, width = 8, height = 6, properties = {
      title = "Kinesis in / scorer lag (ms) / missed fraud", region = var.region, period = 60, view = "timeSeries",
      metrics = [["AWS/Kinesis", "IncomingRecords", "StreamName", aws_kinesis_stream.this["transactions"].name, { stat = "Sum" }],
        ["AWS/Kinesis", "GetRecords.IteratorAgeMilliseconds", "StreamName", aws_kinesis_stream.this["transactions"].name, { stat = "Maximum" }],
    [local.cw_namespace, "FraudMissed", "Service", "stream", { stat = "Sum" }]] } },
    { type = "metric", x = 0, y = 8, width = 8, height = 6, properties = {
      title = "Fargate task CPU / memory %", region = var.region, view = "timeSeries", period = 60, stat = "Average",
      metrics = [["AWS/ECS", "CPUUtilization", "ClusterName", local.name, "ServiceName", local.service_name],
    ["AWS/ECS", "MemoryUtilization", "ClusterName", local.name, "ServiceName", local.service_name]] } },
    { type = "metric", x = 8, y = 8, width = 8, height = 6, properties = {
      title = "Audit archive (Firehose → S3)", region = var.region, view = "timeSeries", period = 300, stat = "Sum",
      metrics = [["AWS/Firehose", "DeliveryToS3.Records", "DeliveryStreamName", aws_kinesis_firehose_delivery_stream.audit.name],
    ["AWS/Firehose", "IncomingRecords", "DeliveryStreamName", aws_kinesis_firehose_delivery_stream.audit.name]] } },
    { type = "log", x = 16, y = 8, width = 8, height = 6, properties = {
      title = "Warnings and errors", region = var.region, view = "table",
    query = "SOURCE '${aws_cloudwatch_log_group.stream.name}' | fields @timestamp, @message | filter @message like /(?i)(error|warn|exception|traceback)/ | sort @timestamp desc | limit 50" } },
  ] })
}
