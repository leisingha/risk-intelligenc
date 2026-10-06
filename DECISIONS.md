# DECISIONS

Every choice made while building this, with the reasoning. Where the spec said
"choose the option with the fewest new dependencies", that is the tie-breaker used.

## Environment and process

| Decision | Reasoning |
|---|---|
| **Python 3.11 venv**, not the system 3.13 | The pinned `torch==2.4.1` and `numpy==1.26.4` have no 3.13 wheels. 3.11 is the oldest version the spec allows and the best supported by the pinned stack. |
| **H2 answered "no key"** | No OpenAI key was provided. The agent runs on the deterministic rule-based planner and an extractive answer composer. The LLM path (`langchain-openai`) is implemented and switches on when `OPENAI_API_KEY` is set. It has not been exercised against the live API. |
| **Code first, data later** | The build container's network policy blocks `data.sec.gov`, `www.sec.gov` and `huggingface.co`. Every module is written and tested offline on synthetic fixtures. The data-dependent phases run with `./run_pipeline.sh` once those hosts are reachable. Until then RESULTS.md says "Not yet measured" rather than showing invented numbers. |
| **`results.json` → generated RESULTS.md** | Spec rule 9: numbers must come from a file, not from memory. Each phase calls `src/results.py:record()`, which re-renders RESULTS.md, so a typed-in number cannot reach the docs. |
| **Added `accelerate==0.34.2`**, the one dependency outside Section 4 | The spec requires the `transformers` Trainer API, and `Trainer` refuses to run without `accelerate>=0.21` (pipeline run 3 failed on exactly this). 0.34.2 is the release matched to `transformers==4.44.2`/`torch==2.4.1`. The alternative, a hand-written training loop, would have broken the "use the Trainer API" requirement instead. |
| **Data phases run in GitHub Actions** (`pipeline.yml`) | The sandbox can't reach SEC or Hugging Face; GitHub runners can. The SEC contact email comes from a repository secret, never from the repo or workflow inputs. Results are committed back by the workflow. |
| **Docker build verified in CI, not locally** | The sandbox's TLS-intercepting proxy is untrusted inside build containers. The Dockerfile is kept clean (no sandbox CA hacks), and GitHub Actions builds it. |

## Data

| Decision | Reasoning |
|---|---|
| **12 companies × 3 most recent 10-Ks = 36 filings**, banking/energy/technology (4 each) | Inside the 30–40 target. Three years per company lets the corpus show how disclosures change across years (e.g. the SEC's 2023 cybersecurity rule added Item 1C). |
| **CIKs hard-coded** in `src/ingest/edgar.py` | This avoids a ticker→CIK lookup request and its failure mode. 12 constants are easy to audit. |
| **150 ms throttle + 3 retries with exponential backoff; 403 raises immediately with a User-Agent hint** | SEC fair-access policy. A 403 is almost always a User-Agent problem, and retrying it only gets the IP blocked. |
| **Extraction = leaf-block paragraphs, then the largest start→end heading span** | EDGAR HTML repeats every heading in the table of contents, so "the first `Item 1A`" is usually the TOC entry. Picking the start/end pair that encloses the most words rejects the TOC without special cases. There is a standalone "Risk Factors" heading fallback for filers that don't use a literal `Item 1A` heading. Failures are raised with countable reasons (`no_item_1a_heading`, `no_end_heading`, `section_too_short`, `empty_document`) and logged to `data/processed/extraction_log.csv`. Nothing is dropped silently. |
| **Passages packed on paragraph boundaries, 200–400 words**; overlong paragraphs split on sentences | Follows the spec. A short final remnant is merged into the previous passage when it fits and is otherwise kept, and the count outside 200–400 is recorded. |

## Labels and classical NLP

| Decision | Reasoning |
|---|---|
| **Keyword rules with per-category hit thresholds** (2 hits for broad categories, 1 for cyber/climate) | A single passing mention ("…pandemics, cyber attacks and…") shouldn't label a passage. Cyber and climate are rarer and their vocabulary is specific, so one hit is enough. |
| **Sample 300 passages stratified by company** | No single filer (some have 2× the risk-factor text of others) can dominate the labelled set. |
| **`labels.csv` carries `label_source`** (`keyword` / `hand`) | It makes clear how many labels a human actually checked. RESULTS.md reports the count. |
| **Stratify the split on each row's rarest positive label** | scikit-learn can't stratify multi-label targets directly. Stratifying on the rarest label keeps cyber and climate in the test set. Singleton strata are pooled. |
| **Split persisted to `data/labels/split.json`** | The baselines and DistilBERT read the same file, so the comparison uses identical rows. |
| **Custom stopwords: scikit-learn's list *minus* `interest`, `system`, `fire`, `bill`, …, *plus* risk-factor boilerplate** | scikit-learn's list contains "interest", which silently deletes the "interest rate" bigram, the single most important market-risk feature. This was found while testing the agent. |
| **No lemmatization** | The spec says lemmatization-free, and the stack has no lemmatizer. Unigram+bigram TF-IDF with sublinear tf absorbs most inflection. Plural folding (`rates→rate`) is applied **only** for term matching in reranking and grounding, where an exact-token miss causes a visible failure. |
| **LogReg `class_weight="balanced"`, C=4** | Imbalance (cyber and climate are rarer) would otherwise push rare classes toward "never predict". |
| **Gradient Boosting via one-vs-rest, 150 shallow trees** | Native GBM is single-output. The sizes are small so CPU training stays in minutes. |
| **Clustering: TF-IDF → TruncatedSVD(100) → L2-normalise → KMeans(k=6)** | Standard LSA. k=6 mirrors the label count so the clusters can be compared against the categories, and silhouette is recorded. |

## Transformer

| Decision | Reasoning |
|---|---|
| **`problem_type="multi_label_classification"`**, float labels, sigmoid + 0.5 threshold | BCEWithLogitsLoss, one independent probability per class. Softmax would force the six to sum to 1 and make "cyber *and* regulatory" impossible to express. A test asserts that the baseline's probabilities can sum above 1. |
| **Positive-class weighting in the BCE loss** (`pos_weight` = neg/pos per class, capped at 20) | The first real fine-tune collapsed: with ~80% negative labels and only 90 optimiser steps at the spec's fixed 3 epochs / batch 8 / lr 2e-5, every sigmoid stayed below 0.5 and the model predicted almost nothing (macro-F1 ≈ 0.02, [pipeline run 4](https://github.com/leisingha/risk-intelligenc/actions/runs/37508555390)). Reporting that as "the baseline wins" would be misleading, since it is an optimisation failure, not a model comparison. Weighting positives rebalances the loss without touching the fixed hyperparameters. Mean predicted probability and predicted-vs-true positive rates per class are now recorded, so a collapse is visible in RESULTS.md. |
| **Fixed 0.5 threshold** (no per-class tuning) | 300 labels leaves no room for a separate validation split. Tuning thresholds on the test set would inflate the comparison. |
| **`save_strategy="no"`, `eval_strategy="no"`** | No validation set (see above). Saves disk. The spec fixes 3 epochs. |
| **Zero-shot BART-MNLI (Phase 3b) skipped** | Optional in the spec, and a 1.6 GB download for a CPU-only comparison. |
| **Classifier fallback to the TF-IDF LogReg** when no DistilBERT is saved | The API, agent and CI work before (or without) Phase 3. `/health` reports which model is live. |

## RAG

| Decision | Reasoning |
|---|---|
| **Chunk size 200 tokens, overlap 32** | all-MiniLM-L6-v2 truncates at 256 word pieces. Embedding a whole 400-word passage would ignore its second half. Chunks keep `passage_id`, and retrieval collapses chunks back to passages (best chunk wins). |
| **Company and sector included in the embedded text; ids, dates and predicted labels excluded** | "Exxon climate" queries benefit from the company name in the vector. Ids are noise to an embedding. |
| **Predicted categories stored as `categories` (string) + `cat_<x>` booleans** | Chroma metadata values must be scalars. The booleans make category filters possible. |
| **Reranker = 0.7·dense + 0.3·idf-weighted query-term coverage** | No new model download (a cross-encoder would add one). Dense finds the topic; coverage rewards chunks that contain the specific terms asked about. Both variants are evaluated. |
| **Embedding backend recorded in collection metadata; a mismatch raises** | Querying an index with a different embedding model silently returns garbage. |
| **Offline `hashing` embedding backend** | CI must not download models. Used only by tests and never for reported metrics. |
| **Retrieval eval written as rules (ticker + required terms)** | The questions were written before the corpus could be downloaded, so passage ids couldn't be known. Each rule resolves to concrete ids against `passages.parquet`. A rule that matches nothing raises an error instead of counting as a miss. After ingestion, pin each item to hand-checked ids (`expected_passage_ids` takes precedence). |
| **Chroma** over PGVector/Weaviate | Embedded mode with a `persist_directory` needs zero infrastructure for ~2k chunks, and the same code talks to a Chroma server in docker-compose. At 10M vectors you'd want a server with sharding, replication, filtered-HNSW performance guarantees and backups (PGVector if the data already lives in Postgres, otherwise Weaviate/Qdrant/OpenSearch). You'd also re-tune HNSW `ef`/`M` and batch the ingestion. |

## Agent and guardrails

| Decision | Reasoning |
|---|---|
| **Node names fixed by the spec; state keys renamed** (`planned_calls`, `final_answer`) | LangGraph forbids a node and a state key sharing a name. |
| **Iteration cap = 5 tool calls**, enforced in `reflect` and the router, plus a LangGraph recursion limit | Two independent stops so a loop can't run away. |
| **One broadened retry when evidence is weak** (drop sector filter, query with category keywords) | Cheap recovery for over-filtered queries. One retry only, so "nothing relevant exists" still ends in a refusal. |
| **Three guardrail layers: scope → relevance → grounding** | Scope catches what retrieval can't fix (unknown companies, post-filing dates, speculation). Relevance catches "nearest neighbour exists but isn't close". Grounding catches drafts that drift from the passages. |
| **Rule-based answers are extractive verbatim quotes, each with a `[passage_id]`** | A quote can be verified exactly (substring check), not approximately. |
| **A quoted sentence must contain ≥ 50% of the question's topic words** | Found while testing: grounding alone allowed an answer that was perfectly cited but off-topic ("key personnel" quoted for a cyber question). Grounded ≠ relevant. |
| **Unknown-company detection = capitalised spans not matching a corpus alias** | No NER model in the stack. Known gap: a lowercase or unusual company name slips through to retrieval, where the relevance and grounding layers are the backstop. |

## What I would do differently with more time

1. **More labels and a validation split.** 300 is enough to compare models but not to tune thresholds or report confidence intervals. I'd bootstrap CIs on macro-F1 and label more cyber/climate passages.
2. **A cross-encoder reranker** (e.g. ms-marco-MiniLM), evaluated against the lexical reranker on the same 20 questions, and a larger eval set (20 questions gives very coarse hit-rate steps: 0.05 per question).
3. **NLI-based grounding** (claim ⇒ entailed by passage) instead of lexical support for LLM paraphrases. Lexical support can't see negation.
4. **NER for the scope check**, instead of the capitalisation heuristic.
5. **Year-aware retrieval** (filter or boost by filing year), since the corpus has three filings per company.
6. **Strip company self-references from classifier features.** The first real run shows `jpmorganchase` among the top LogReg features for *credit*: the model partly learns "which company" as a proxy for "which risk", which won't transfer to new filers.
7. **Hand-pinned retrieval answers** for every eval item, replacing the rules.
