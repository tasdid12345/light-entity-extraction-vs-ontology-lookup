# Light entity extraction vs. ontology lookup (species_800)

Question: can a lightweight context-aware token classifier with ontology-term features extract organism entities from scientific text better than a plain ontology-term lookup?

- Dataset: species_800 (official train/test split), downloaded from the Hugging Face Hub.
- Baseline: longest-match dictionary lookup using training entity strings.
- Improved: logistic-regression BIO tagger (word, affix, shape, context and ontology-match features); C tuned on a 10% held-out part of the training set.
- Metric: entity-level exact-match F1 on the test set (higher is better).
- Seed 42, CPU only.

Run `python analysis.py`; it writes `results.json` and `figures/f1.png`.

Fix note: test sentences with zero tokens previously crashed the tagger's vectorizer; they now get empty predictions (the baseline already handled them the same way).

## Results (latest run)

On the species_800 test set, entity-level exact-match F1 was 30% for a standard ontology-term dictionary lookup and 44% for a logistic-regression BIO tagger with ontology-match features, a gain on this biomedical stand-in dataset rather than agricultural text.

| Metric | Value |
|---|---|
| baseline_score | 0.3018 |
| improved_score | 0.4432 |
| improvement | 0.1414 |
| baseline_f1 | 0.3018 |
| improved_f1 | 0.4432 |
| baseline_precision | 0.2642 |
| baseline_recall | 0.352 |
| improved_precision | 0.5436 |
| improved_recall | 0.3742 |
| best_C | 0.1 |
| dev_f1_baseline | 0.5589 |
| dev_f1_improved | 0.752 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Deterministic models (single run, seed 42). The LLM of the paper is not re-implemented; species_800 (biomedical abstracts) is a stand-in for agricultural text. The ontology list comes from training annotations, so lookup recall is bounded by training vocabulary. Tagger ontology features use out-of-fold dictionaries on training data. Data loading fixed: the Hub id 'species_800' returned 401, so namespaced ids are tried. Empty test sentences (zero tokens) are passed through with empty predictions in both models instead of crashing the vectorizer.

## Reference

This project is motivated by the following paper:

> S Wilson, A Ginige, J Goonatilake (2024). *Design science research approach for ontology development in agriculture: utilising advances of LLM for automated entity extraction*. aisel.aisnet.org. https://aisel.aisnet.org/acis2024/150/

Paper author: Prof. Athula Ginige, Western Sydney University (a.ginige@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
