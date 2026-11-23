# Amazon Kinesis Data Streams: the event bus. One shard each (1 MB/s or 1,000 records/s in: plenty for the demo;
# production scales shards with the 2,800 TPS peak). KMS-encrypted with the AWS managed key; 24 h retention.
#   <name>-transactions  producer -> stream scorer
#   <name>-decisions     every scored transaction with its action (APPROVE / APPROVE_STEP_UP / HOLD_FOR_REVIEW /
#                        BLOCK_AND_ALERT): the hook for downstream systems; Firehose archives it to the audit bucket,
#                        and the analytics store is rebuilt from it after a restart

locals {
  streams = { transactions = 24, decisions = 24 } # name => retention hours
}

resource "aws_kinesis_stream" "this" {
  for_each         = local.streams
  name             = "${local.name}-${each.key}"
  retention_period = each.value
  shard_count      = 1
  encryption_type  = "KMS"
  kms_key_id       = "alias/aws/kinesis"
  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
}
