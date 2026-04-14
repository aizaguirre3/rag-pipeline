# RAG Pipeline Results Report

End-to-end evaluation of the RAG pipeline on enterprise MLOps documentation. All results generated with live API calls against the Anthropic Claude API.

## Knowledge Base

| Document | Chunks | Content |
|----------|--------|---------|
| platform_api_docs.txt | 8 | Authentication, endpoints, rate limits, webhooks |
| incident_reports.txt | 16 | 4 production incidents (latency, vector DB, prompt injection, cost overrun) |
| system_architecture.txt | 10 | Infrastructure, RAG pipeline design, caching, disaster recovery |
| ml_ops_runbook.txt | 3 | Deployment procedures, autoscaling, alerting, rollback |
| data_governance_policy.txt | 6 | PII handling, data classification, compliance requirements |
| prompt_engineering.txt | 7 | System prompts, few-shot, chain-of-thought, structured output |
| rag_architecture.txt | 8 | RAG stages, chunking strategies, failure modes, evaluation |
| llm_fundamentals.txt | 7 | Transformer architecture, training, tokenization, scaling laws |
| **Total** | **65 chunks** | **8 documents** |

## Batch Query Results (15 Industry Questions)

### Pipeline Performance

| Metric | Value |
|--------|-------|
| **Questions Answered** | 15/15 |
| **Model** | Claude Sonnet 4 |
| **Avg Latency** | 4,273 ms |
| **Total Input Tokens** | 14,043 |
| **Total Output Tokens** | 2,986 |
| **Chunks Retrieved Per Query** | 5 |
| **Avg Top Chunk Relevance** | 0.85 |

### Per-Question Results

| # | Question | Sources | Latency | Tokens (in/out) |
|---|----------|---------|---------|-----------------|
| 1 | API rate limits for enterprise tier | platform_api_docs.txt | 3,326ms | 736/100 |
| 2 | Access token duration and refresh | platform_api_docs.txt | 2,282ms | 698/127 |
| 3 | Root cause of latency spike incident | incident_reports.txt | 2,992ms | 1,022/198 |
| 4 | Vector DB inconsistency cause and resolution | incident_reports.txt | 5,253ms | 1,103/293 |
| 5 | Canary deployment procedure | ml_ops_runbook.txt | 6,727ms | 909/326 |
| 6 | Instance type for 13B model | ml_ops_runbook.txt | 3,355ms | 886/132 |
| 7 | Autoscaling triggers | ml_ops_runbook.txt | 3,095ms | 866/195 |
| 8 | PII handling in ML pipelines | data_governance_policy.txt | 5,622ms | 983/259 |
| 9 | LLM prompt/response logging policy | data_governance_policy.txt | 3,433ms | 965/158 |
| 10 | RAG caching strategy for cost reduction | system_architecture.txt | 2,790ms | 1,009/145 |
| 11 | Hybrid retrieval system architecture | system_architecture.txt, rag_architecture.txt | 6,172ms | 1,078/304 |
| 12 | Token explosion financial impact | incident_reports.txt | 4,424ms | 1,002/238 |
| 13 | Data classification tiers for PII | data_governance_policy.txt | 2,771ms | 886/119 |
| 14 | Critical PagerDuty alert thresholds | ml_ops_runbook.txt | 3,337ms | 908/152 |
| 15 | Prompt injection incident response | incident_reports.txt | 8,511ms | 992/340 |

## Pipeline Architecture in Action

```
User Question
    │
    ▼
┌──────────────────────────────────────────┐
│  Retrieval (ChromaDB)                    │
│  Query → Embedding → Cosine Similarity   │
│  → Top 5 chunks (avg relevance: 0.85)   │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Generation (Claude Sonnet 4)            │
│  System prompt enforces:                 │
│  - Answer ONLY from provided context     │
│  - Cite sources                          │
│  - Say "I don't know" if insufficient    │
│  Avg latency: 4,273ms                   │
└──────────────────────────────────────────┘
    │
    ▼
  Answer with source attribution
```

## Integration with LLM Eval Harness

The RAG pipeline's outputs can be scored by the [LLM Eval Harness](https://github.com/aizaguirre3/llm-eval-harness) using RAGAS metrics. When evaluated against ground truth answers:

| Metric | Sonnet 4 | Haiku 4.5 |
|--------|----------|-----------|
| **Faithfulness** | 0.671 | **0.817** |
| **Context Precision** | **1.000** | **1.000** |
| **Context Recall** | **1.000** | **1.000** |
| **Avg Latency** | 5,015 ms | **2,412 ms** |

This creates a complete **build-measure-optimize** loop:
1. **Build**: RAG pipeline ingests documents and generates answers
2. **Measure**: Eval harness scores answers with RAGAS + custom metrics
3. **Optimize**: Regression testing catches quality drops when changing chunking, retrieval, or prompts

## Observations

1. **Source attribution works correctly** -- every answer cites the relevant source documents
2. **Cross-document retrieval** succeeds when questions span multiple topics (e.g., hybrid retrieval pulls from both system_architecture.txt and rag_architecture.txt)
3. **Grounding constraints are respected** -- the model acknowledges when context is insufficient (e.g., token refresh details)
4. **Chunking quality matters** -- questions about incident reports score well because the sentence-aware chunker keeps incident narratives coherent
5. **Latency correlates with answer complexity** -- simple factual lookups (2-3s) vs multi-part incident analysis (5-8s)

---

*Report generated April 14, 2026. All results from live Anthropic Claude API calls.*
