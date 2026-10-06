#!/usr/bin/env bash
# Run every data-dependent phase in order, stopping on the first failure.
# Needs network access to data.sec.gov, www.sec.gov and huggingface.co, and
# SEC_CONTACT_EMAIL in .env. Usage: ./run_pipeline.sh [from_phase]   (default 0)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
FROM=${1:-0}

step() { echo; echo "=== $* ==="; }

if [ "$FROM" -le 0 ]; then
  step "Phase 0: EDGAR smoke test"
  $PY -m src.ingest.edgar --smoke --tickers JPM
fi
if [ "$FROM" -le 1 ]; then
  step "Phase 1: fetch 3 x 10-K for 12 companies, extract Item 1A"
  $PY -m src.ingest.edgar --per-company 3
fi
if [ "$FROM" -le 2 ]; then
  if [ ! -f data/labels/labels.csv ]; then
    step "Phase 2: bootstrap labels"
    $PY -m src.nlp.labels --bootstrap
    echo
    echo "=== HUMAN STEP ==="
    echo "Hand-correct data/labels/labels.csv (set label_source=hand per reviewed row),"
    echo "then re-run: ./run_pipeline.sh 2"
    exit 0
  fi
  step "Phase 2: label stats, baselines, clustering"
  $PY -m src.nlp.labels --stats
  $PY -m src.nlp.baseline
  $PY -m src.nlp.cluster
fi
if [ "$FROM" -le 3 ]; then
  step "Phase 3: fine-tune DistilBERT, then verify reproducibility"
  $PY -m src.models.finetune
  $PY -m src.models.finetune --eval-only
fi
if [ "$FROM" -le 4 ]; then
  step "Phase 4: build Chroma index, evaluate retrieval"
  $PY -m src.rag.index
  $PY -m src.rag.retrieve --eval
fi
if [ "$FROM" -le 5 ]; then
  step "Phase 5: agent demo + adversarial set"
  $PY -m src.agent.graph --demo --adversarial
  echo "Review data/eval/adversarial_transcript.md (HUMAN GATE H3)."
fi
step "Figures"
$PY -m src.report
echo "Done. Numbers are in RESULTS.md."
