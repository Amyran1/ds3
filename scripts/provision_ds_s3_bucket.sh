#!/usr/bin/env bash
# Provision the DS artifacts S3 bucket (Chorus org account).
set -euo pipefail

BUCKET="${BUCKET:-chorus-ds-artifacts}"
REGION="${REGION:-us-east-1}"

echo "Creating s3://${BUCKET} in ${REGION}..."
aws s3 mb "s3://${BUCKET}" --region "${REGION}"

echo "Blocking public access..."
aws s3api put-public-access-block \
  --bucket "${BUCKET}" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "Enabling default encryption (SSE-S3)..."
aws s3api put-bucket-encryption \
  --bucket "${BUCKET}" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

echo "Enabling versioning..."
aws s3api put-bucket-versioning \
  --bucket "${BUCKET}" \
  --versioning-configuration Status=Enabled

LIFECYCLE="$(mktemp)"
trap 'rm -f "${LIFECYCLE}"' EXIT
cat > "${LIFECYCLE}" <<'EOF'
{
  "Rules": [{
    "ID": "expire-scratch-prefix-90d",
    "Status": "Enabled",
    "Filter": { "Prefix": "autonomous-data-scientist/scratch/" },
    "Expiration": { "Days": 90 }
  }]
}
EOF

echo "Applying scratch-prefix lifecycle rule..."
aws s3api put-bucket-lifecycle-configuration \
  --bucket "${BUCKET}" \
  --lifecycle-configuration "file://${LIFECYCLE}"

echo "Smoke check..."
echo "ok" | aws s3 cp - "s3://${BUCKET}/_healthcheck/probe.txt"
aws s3 ls "s3://${BUCKET}/"
aws s3 rm "s3://${BUCKET}/_healthcheck/probe.txt"

echo "Done: s3://${BUCKET}"
