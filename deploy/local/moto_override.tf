# Rehearsal only (deploy/local/rehearse.py copies this next to a COPY of deploy/terraform, inside a Docker volume):
# the same stack, applied to Moto (an AWS emulator in a container) with local state. Never used against AWS; the real
# Terraform files stay untouched and nothing is written into the repository.
terraform {
  backend "local" {
    path = "/work/state/terraform.tfstate"
  }
}

provider "aws" {
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
  s3_use_path_style           = true
  endpoints {
    cloudwatch = "http://aws:5000"
    ec2        = "http://aws:5000"
    ecr        = "http://aws:5000"
    ecs        = "http://aws:5000"
    elbv2      = "http://aws:5000"
    firehose   = "http://aws:5000"
    iam        = "http://aws:5000"
    kinesis    = "http://aws:5000"
    kms        = "http://aws:5000"
    logs       = "http://aws:5000"
    s3         = "http://aws:5000"
    sns        = "http://aws:5000"
    sqs        = "http://aws:5000"
    ssm        = "http://aws:5000"
    sts        = "http://aws:5000"
  }
}
