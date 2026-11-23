locals {
  base_url = "${local.https ? "https" : "http"}://${aws_lb.this.dns_name}"
}

output "stream_url" {
  description = "ARGUS_STREAM_URL for the platform (on the GPU VM or your PC): the UI's Live stream page uses it"
  value       = local.base_url
}
output "stream_api_docs" { value = "${local.base_url}/live/docs" }
output "control_key_parameter" {
  description = "SSM parameter holding ARGUS_STREAM_CONTROL_KEY (ops.py platform-env reads it)"
  value       = aws_ssm_parameter.secret["control-key"].name
}
output "cloudwatch_dashboard" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards/dashboard/${local.name}"
}
output "scoring_url" { value = local.scoring_url }
output "kinesis_streams" { value = { for k, s in aws_kinesis_stream.this : k => s.name } }
output "data_bucket" { value = aws_s3_bucket.this["data"].bucket }
output "audit_bucket" { value = aws_s3_bucket.this["audit"].bucket }
output "ecr" { value = aws_ecr_repository.stream.repository_url }
output "cluster" { value = aws_ecs_cluster.this.name }
output "services" { value = [aws_ecs_service.stream.name] }
output "actions_topic_arn" { value = aws_sns_topic.actions.arn }
output "alerts_topic_arn" { value = aws_sns_topic.alerts.arn }
