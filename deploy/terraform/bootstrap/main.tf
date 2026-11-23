# One-time bootstrap, run by `python deploy/ops.py up` (local state in this folder, git-ignored):
#   - S3 bucket for the main stack's Terraform state (versioned, encrypted, private)
#   - GitHub Actions OIDC provider (only if the account has none yet)
#   - IAM role GitHub Actions assumes to deploy / destroy: only from this repo's main branch or its "production"
#     environment, and it may manage IAM only for names starting with "<project>-". No AWS keys live in GitHub.
# Destroy it last: `python deploy/ops.py down --all`.

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.70" }
  }
}

variable "region" {
  type    = string
  default = "ap-south-1"
}
variable "project" {
  type    = string
  default = "argus"
}
variable "github_repo" {
  description = "owner/repo that may deploy, e.g. 6-month-fde-challenge/project_argus ('' = no CI role)"
  type        = string
  default     = ""
}
variable "create_oidc_provider" {
  description = "false when the account already has token.actions.githubusercontent.com (ops.py detects this)"
  type        = bool
  default     = true
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform", Stack = "bootstrap" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  account = data.aws_caller_identity.me.account_id
  ci      = var.github_repo != ""
  oidc_arn = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : (
  "arn:aws:iam::${local.account}:oidc-provider/token.actions.githubusercontent.com")
}

# ---------------------------------------------------------------- state bucket
resource "aws_s3_bucket" "state" {
  bucket        = "${var.project}-tfstate-${local.account}-${var.region}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------- GitHub OIDC + deploy role
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

resource "aws_iam_role" "github" {
  count = local.ci ? 1 : 0
  name  = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        # only jobs of the `production` environment (deploy.yml, destroy.yml); ops.py restricts that environment to
        # the main branch, so a workflow on another branch or a pull request can never assume this role
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:environment:production"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "power" {
  count      = local.ci ? 1 : 0
  role       = aws_iam_role.github[0].name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

resource "aws_iam_role_policy" "iam_scoped" {
  count = local.ci ? 1 : 0
  name  = "${var.project}-iam-scoped"
  role  = aws_iam_role.github[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ManageOnlyProjectIam"
        Effect = "Allow"
        Action = "iam:*"
        Resource = [
          "arn:aws:iam::${local.account}:role/${var.project}-*",
          "arn:aws:iam::${local.account}:policy/${var.project}-*",
        ]
      },
      { Sid = "ReadIam", Effect = "Allow", Action = ["iam:Get*", "iam:List*"], Resource = "*" },
      { Sid = "ServiceLinkedRoles", Effect = "Allow", Action = "iam:CreateServiceLinkedRole", Resource = "*" },
      {
        # the deploy role matches "${var.project}-*" too: it must never change its own permissions or trust
        Sid      = "NeverChangeItself"
        Effect   = "Deny"
        Action   = ["iam:Put*", "iam:Attach*", "iam:Detach*", "iam:Delete*", "iam:Update*", "iam:Tag*", "iam:Untag*"]
        Resource = "arn:aws:iam::${local.account}:role/${var.project}-github-deploy"
      },
      {
        # project roles get the project's own policies and the one AWS-managed policy the stack uses, never more
        Sid      = "OnlyProjectOrEcsPolicies"
        Effect   = "Deny"
        Action   = "iam:AttachRolePolicy"
        Resource = "*"
        Condition = { ArnNotLike = { "iam:PolicyARN" = [
          "arn:aws:iam::${local.account}:policy/${var.project}-*",
          "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
        ] } }
      },
    ]
  })
}

output "state_bucket" { value = aws_s3_bucket.state.bucket }
output "github_role_arn" { value = local.ci ? aws_iam_role.github[0].arn : "" }
output "region" { value = var.region }
