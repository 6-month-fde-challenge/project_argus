# Own VPC (nothing shared with other projects in the account), 2 public subnets (the ALB needs two AZs), no NAT.
# The Fargate task gets a public IP for its outbound calls (ECR, Kinesis, the GPU VM); its security group accepts
# traffic only from the ALB, and the ALB only from allowed_cidrs.

data "aws_availability_zones" "available" { state = "available" }

locals {
  name  = "${var.project}-${var.environment}"
  tags  = { Project = var.project, Environment = var.environment, ManagedBy = "terraform" }
  azs   = slice(data.aws_availability_zones.available.names, 0, 2)
  https = var.certificate_arn != ""
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  tags                 = { Name = local.name }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = local.name }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(aws_vpc.this.cidr_block, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name}-public-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = { Name = "${local.name}-public" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------- security groups
resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "stream API from allowed_cidrs"
  vpc_id      = aws_vpc.this.id
  dynamic "ingress" {
    for_each = local.https ? [80, 443] : [80]
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidrs
    }
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name        = "${local.name}-app"
  description = "stream task: from the ALB only"
  vpc_id      = aws_vpc.this.id
  ingress {
    from_port       = 8003
    to_port         = 8003
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------- load balancer (everything is under /live)
resource "aws_lb" "this" {
  name            = local.name
  subnets         = aws_subnet.public[*].id
  security_groups = [aws_security_group.alb.id]
  idle_timeout    = 120
}

resource "aws_lb_target_group" "stream" {
  name                 = "${local.name}-stream"
  port                 = 8003
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.this.id
  deregistration_delay = 10
  health_check {
    path    = "/live/health"
    matcher = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = local.https ? "redirect" : "forward"
    target_group_arn = local.https ? null : aws_lb_target_group.stream.arn
    dynamic "redirect" {
      for_each = local.https ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = local.https ? 1 : 0
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.stream.arn
  }
}
