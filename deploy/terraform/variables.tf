variable "region" {
  description = "CRD: no data, prompts or weights leave VKPayments' AWS account in ap-south-1 (Mumbai)"
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Prefix of every resource name; the CI role may only manage IAM named <project>-*"
  type        = string
  default     = "argus"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "vpc_cidr" {
  description = "Own VPC, shared with nothing (no peering)"
  type        = string
  default     = "10.50.0.0/16"
}

variable "image_tag" {
  description = "Tag of the stream image in ECR (CI passes the git SHA)"
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------- the Fraud Model API (GPU VM, deploy/hyperstack)
variable "scoring_host" {
  description = "Public IP / DNS of the GPU VM serving the Fraud Model API (deploy/hyperstack/serve.sh)"
  type        = string
  default     = ""
}

variable "scoring_port" {
  type    = number
  default = 8080
}

variable "scoring_scheme" {
  description = "http, or https once a TLS proxy with a certificate fronts the Fraud Model API (deploy/README.md section 7)"
  type        = string
  default     = "http"
  validation {
    condition     = contains(["http", "https"], var.scoring_scheme)
    error_message = "scoring_scheme must be http or https."
  }
}

variable "scoring_token" {
  description = "ARGUS_SCORING_TOKEN from the VM's .env (stored in SSM, never in the image or in git)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "scoring_timeout_s" {
  description = "Client timeout per scoring call; past it the fail-open rule decides (FR-API-01)"
  type        = number
  default     = 5
}

variable "scoring_deadline_ms" {
  description = "X-Request-Deadline-Ms sent with every call (LLD 6.3); 0 = the API's own budget"
  type        = number
  default     = 0
}

# ---------------------------------------------------------------- traffic
variable "events_per_sec" {
  description = "Default traffic rate (changeable live on the UI's Live stream page)"
  type        = number
  default     = 2
}

variable "fraud_rate" {
  type    = number
  default = 0.04
}

variable "producer_source" {
  description = "synthetic | holdout | mix (holdout replays the labelled Jan-Feb 2026 split from the data bucket)"
  type        = string
  default     = "mix"
  validation {
    condition     = contains(["synthetic", "holdout", "mix"], var.producer_source)
    error_message = "producer_source must be synthetic, holdout or mix."
  }
}

variable "transaction_time" {
  description = "pattern: synthetic fraud at its typical hour (API on local_demo) | now: always now (API on prod_a100, +-15 min window)"
  type        = string
  default     = "pattern"
  validation {
    condition     = contains(["pattern", "now"], var.transaction_time)
    error_message = "transaction_time must be pattern or now."
  }
}

# ---------------------------------------------------------------- Fargate task
variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

# ---------------------------------------------------------------- access, audit, ops
variable "allowed_cidrs" {
  description = "Who may reach the stream API (ops.py sets: the GPU VM, where the platform runs, + your current IP)"
  type        = list(string)
  # no default: an open ALB must be a deliberate choice (["0.0.0.0/0"]), never what an unset variable falls back to
  validation {
    condition     = length(var.allowed_cidrs) > 0 && alltrue([for c in var.allowed_cidrs : can(cidrhost(c, 0))])
    error_message = "allowed_cidrs needs at least one valid CIDR, e.g. [\"203.0.113.7/32\"] (ops.py fills it in)."
  }
}

variable "certificate_arn" {
  description = "Optional ACM certificate for HTTPS on the load balancer ('' = HTTP only)"
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Optional e-mail for CloudWatch alarms (confirm the SNS subscription mail)"
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 14
}

variable "audit_retention_days" {
  description = "Decisions archived in S3 (NFR-11: 5 years). Glacier after 90 days."
  type        = number
  default     = 1827
}
