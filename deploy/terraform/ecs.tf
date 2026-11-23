# ECR + secrets + IAM + ONE ECS Fargate service running the stream container (no EC2 instances to manage):
#   stream :8003  Kinesis producer (off until started from the UI) + scorer calling the Fraud Model API on the GPU VM
#                 + analytics API under /live (the ARGUS platform proxies it for the UI's Live stream page)
# The models are NOT on AWS here: they run on the GPU VM (deploy/hyperstack); the stream calls them over HTTPS/HTTP
# with the service token from SSM.

resource "aws_ecr_repository" "stream" {
  name                 = "${local.name}/stream"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # `ops.py down` removes the images too
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "stream" {
  repository = aws_ecr_repository.stream.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "keep the last 10 images"
    selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
    action       = { type = "expire" }
  }] })
}

# ---------------------------------------------------------------- secrets (SSM SecureString, injected by ECS)
resource "random_password" "control_key" {
  length  = 40
  special = false
}

locals {
  secrets = {
    scoring-token = var.scoring_token == "" ? "unset" : var.scoring_token
    control-key   = random_password.control_key.result
  }
}

resource "aws_ssm_parameter" "secret" {
  for_each = local.secrets
  name     = "/${local.name}/${each.key}"
  type     = "SecureString"
  value    = each.value
}

# ---------------------------------------------------------------- IAM
data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "${local.name}-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = ["ssm:GetParameters"], Resource = [for p in aws_ssm_parameter.secret : p.arn]
  }] })
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy" "task" {
  name = "${local.name}-task"
  role = aws_iam_role.task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.this["data"].arn },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.this["data"].arn}/*" },
    { Effect = "Allow", Action = ["sns:Publish"], Resource = aws_sns_topic.actions.arn },
    { Effect = "Allow", Resource = [for s in aws_kinesis_stream.this : s.arn], Action = [
      "kinesis:PutRecord", "kinesis:PutRecords", "kinesis:GetRecords", "kinesis:GetShardIterator",
    "kinesis:DescribeStreamSummary", "kinesis:DescribeStream", "kinesis:ListShards"] },
    { Effect = "Allow", Action = ["kms:GenerateDataKey", "kms:Decrypt"], Resource = "*",
    Condition = { StringEquals = { "kms:ViaService" = "kinesis.${var.region}.amazonaws.com" } } },
    { Effect = "Allow", Action = ["cloudwatch:PutMetricData"], Resource = "*",
    Condition = { StringEquals = { "cloudwatch:namespace" = local.cw_namespace } } },
    { Effect = "Allow", Resource = "*", Action = [ # ECS Exec (aws ecs execute-command) for debugging
    "ssmmessages:CreateControlChannel", "ssmmessages:CreateDataChannel", "ssmmessages:OpenControlChannel", "ssmmessages:OpenDataChannel"] },
  ] })
}

# ---------------------------------------------------------------- cluster, task, service
resource "aws_cloudwatch_log_group" "container_insights" { # created up front so `destroy` removes it too
  name              = "/aws/ecs/containerinsights/${local.name}/performance"
  retention_in_days = 1
}

resource "aws_ecs_cluster" "this" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  depends_on = [aws_cloudwatch_log_group.container_insights]
}

locals {
  cw_namespace = "Argus/${var.environment}"
  service_name = "${local.name}-stream"
  scoring_url  = var.scoring_host == "" ? "" : "${var.scoring_scheme}://${var.scoring_host}:${var.scoring_port}"
  stream_env = {
    AWS_REGION          = var.region, KINESIS_ENABLED = "1"
    STREAM_TRANSACTIONS = aws_kinesis_stream.this["transactions"].name
    STREAM_DECISIONS    = aws_kinesis_stream.this["decisions"].name
    SCORING_URL         = local.scoring_url, SCORING_TIMEOUT_S = tostring(var.scoring_timeout_s)
    SCORING_DEADLINE_MS = tostring(var.scoring_deadline_ms)
    CW_NAMESPACE        = local.cw_namespace, SNS_ACTIONS_TOPIC_ARN = aws_sns_topic.actions.arn
    STORE_PATH          = "/data/argus_stream.db", AUTO_START = "0"
    EVENTS_PER_SEC      = tostring(var.events_per_sec), FRAUD_RATE = tostring(var.fraud_rate)
    PRODUCER_SOURCE     = var.producer_source, TRANSACTION_TIME = var.transaction_time
    DATASET_URI         = "s3://${aws_s3_bucket.this["data"].bucket}/holdout/"
  }
}

resource "aws_ecs_task_definition" "stream" {
  family                   = local.service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name         = "stream"
    image        = "${aws_ecr_repository.stream.repository_url}:${var.image_tag}"
    essential    = true
    portMappings = [{ containerPort = 8003, protocol = "tcp" }]
    environment  = [for k, v in local.stream_env : { name = k, value = v }]
    secrets = [{ name = "ARGUS_SCORING_TOKEN", valueFrom = aws_ssm_parameter.secret["scoring-token"].arn },
    { name = "CONTROL_API_KEY", valueFrom = aws_ssm_parameter.secret["control-key"].arn }]
    linuxParameters = { initProcessEnabled = true }
    stopTimeout     = 30
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8003/live/health')\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = { awslogs-group = aws_cloudwatch_log_group.stream.name, awslogs-region = var.region,
      awslogs-stream-prefix = "stream" }
    }
  }])
}

resource "aws_ecs_service" "stream" {
  name            = local.service_name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.stream.arn
  desired_count   = 1 # one scorer owns the analytics store (rebuilt from the decisions stream on restart)
  launch_type     = "FARGATE"
  # stop the old task before the new one starts: two scorers never write analytics at the same time
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  enable_execute_command             = true
  health_check_grace_period_seconds  = 120
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true # ECR, Kinesis and the GPU VM without a NAT gateway
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.stream.arn
    container_name   = "stream"
    container_port   = 8003
  }
  depends_on = [aws_lb_listener.http, aws_iam_role_policy.task, aws_kinesis_stream.this]
}
