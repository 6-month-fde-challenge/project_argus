# ARGUS on AWS (ap-south-1): the real-time stream around the Fraud Model API. See deploy/README.md.
terraform {
  required_version = ">= 1.10" # S3-native state locking (use_lockfile)
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 6.70" }
    random = { source = "hashicorp/random", version = "~> 3.9" }
  }
  # bucket / key / region come from deploy/ops.py (the state bucket is created by deploy/terraform/bootstrap)
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags { tags = local.tags } # every resource is tagged Project=<project>: `ops.py audit` finds them all
}
