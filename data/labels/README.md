# Labelled dataset

`labels.csv` holds 300 passages labelled for six risk categories (multi-label 0/1
columns: credit, market, operational, regulatory, cyber, climate), plus the passage
text so the file is self-contained, and `label_source` (`keyword` = rule bootstrap,
`hand` = human-corrected).

`split.json` records the fixed train/test passage ids (seed 42) shared by the
baselines and DistilBERT so every model is evaluated on identical rows.

Produce them with:

    python -m src.nlp.labels --bootstrap   # keyword rules, writes labels.csv
    # hand-correct labels.csv, setting label_source=hand on each reviewed row
    python -m src.nlp.labels --stats       # records counts in RESULTS.md
    python -m src.nlp.baseline             # creates split.json on first run
