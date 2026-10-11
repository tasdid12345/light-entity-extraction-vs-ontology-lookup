# Light entity extraction vs. ontology lookup (species_800)

Question: can a lightweight context-aware token classifier with ontology-term features extract organism entities from scientific text better than a plain ontology-term lookup?

- Dataset: species_800 (official train/test split), downloaded from the Hugging Face Hub.
- Baseline: longest-match dictionary lookup using training entity strings.
- Improved: logistic-regression BIO tagger (word, affix, shape, context and ontology-match features); C tuned on a 10% held-out part of the training set.
- Metric: entity-level exact-match F1 on the test set (higher is better).
- Seed 42, CPU only.

Run `python analysis.py`; it writes `results.json` and `figures/f1.png`.

Fix note: test sentences with zero tokens previously crashed the tagger's vectorizer; they now get empty predictions (the baseline already handled them the same way).

## Reference

This project is motivated by the following paper:

> S Wilson, A Ginige, J Goonatilake (2024). *Design science research approach for ontology development in agriculture: utilising advances of LLM for automated entity extraction*. aisel.aisnet.org. https://aisel.aisnet.org/acis2024/150/

Paper author: Prof. Athula Ginige, Western Sydney University (a.ginige@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
