# Data Quality Report

## Scope and protection

The existing-company exclusion set was recovered from the specified backend repository's own historical `data/` tree and normalized with aliases. The new-company list is disjoint from that set and from the ten mandatory exclusions in the research brief. No existing records were deleted or overwritten.

## Counts

| Metric | Value |
|---|---:|
| Existing companies detected | 29 |
| New companies added | 30 |
| Total structured records | 122 |
| Reported questions | 21 |
| Reported interview-process records | 28 |
| Inferred questions | 30 |
| Role-derived records | 13 |
| Company-technical-focus records | 30 |
| Unique source domains | 30 |
| Unique source URLs in JSONL | 51 |
| Date range of exposed dates | 2021-11-01 to 2026-08-14 |
| Duplicate records removed | 0 |

## New companies

OpenAI, Anthropic, Cohere, Perplexity, xAI, Databricks, Snowflake, Scale AI, LangChain, Replit, Weights & Biases, Harvey, Runway, Glean, Together AI, Groq, Salesforce, ServiceNow, Palantir, Stripe, Atlassian, AMD, Intel, Qualcomm, PayPal, Mastercard, Spotify, ByteDance, Tencent, Alibaba

## Weak or incomplete evidence

Databricks, Glean, Groq, Mastercard, Palantir, Runway, ServiceNow, Snowflake, Spotify, Together AI, Weights & Biases, xAI

Companies with low-confidence entries are retained because the brief requests broad coverage, but those entries are explicitly marked and should not be used as strong claims. Several companies have strong technical-context evidence but limited public candidate-reported AI/ML questions. No generic interview question was relabeled as a reported company question.

## Integration status

The backend's ingestion code supports Markdown documents and recursively scans `data/`. The generated per-company Markdown documents are saved under `data/new_company_interview_research/` and were successfully processed by the existing ingestion CLI into a separate 31-chunk verification index. The current checkout does not contain the historical PDF corpus because the specified repository deleted its `data/` directory in commit `069978e`; the original knowledge base and its production vector index were not replaced or rebuilt.
