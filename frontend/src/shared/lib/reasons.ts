/** Reason codes (CRD 9.3): what support staff read next to a decision. */
export const REASONS: Record<string, { label: string; meaning: string }> = {
  R01_NEW_DEVICE: { label: "New device", meaning: "Transaction from a device not previously used by the customer" },
  R02_GEO_MISMATCH: { label: "Geo mismatch", meaning: "Customer far from merchant or in an unusual city" },
  R03_ODD_HOUR: { label: "Odd hour", meaning: "Transaction at an unusual time of day (midnight - 5 AM)" },
  R04_FAILED_ATTEMPTS: { label: "Failed attempts", meaning: "Multiple failed attempts before this transaction" },
  R05_AMOUNT_SPIKE: { label: "Amount spike", meaning: "Amount much higher than the customer's usual spend" },
  R06_NEW_ACCOUNT: { label: "New account", meaning: "Account created very recently" },
  R07_VELOCITY: { label: "Velocity", meaning: "Unusually many transactions in a short period" },
  R08_HIGH_RISK_MCC: { label: "High-risk MCC", meaning: "Merchant category with elevated fraud history" },
};

export const DECISION_POLICY = [
  { range: "0.00 – 0.29", risk: "LOW", decision: "ALLOW", action: "Approve normally" },
  { range: "0.30 – 0.59", risk: "MEDIUM", decision: "ALLOW", action: "Approve; step-up OTP if supported" },
  { range: "0.60 – 0.84", risk: "HIGH", decision: "REVIEW", action: "Hold, send to manual review" },
  { range: "0.85 – 1.00", risk: "CRITICAL", decision: "BLOCK", action: "Decline and notify the customer" },
];
