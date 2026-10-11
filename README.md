# Context-aware organism tagger vs. ontology lookup on species_800

Motivation: the paper "Design science research approach for ontology development in agriculture: utilising advances of LLM for automated entity extraction" (2024) uses an LLM to extract entities for building ontologies. Re-implementing an LLM is costly, so this experiment uses a cheap, measurable stand-in: does a small context-aware classifier with ontology-term features beat a plain ontology-term lookup?

## Question
Can a lightweight context-aware token classifier with ontology-term features extract organism entities from scientific text better than a plain ontology-term dictionary lookup?

## Dataset
`species_800` (organism mentions in abstracts), downloaded as parquet files from the Hugging Face API, using the official train and test splits. 10% of train is held out for tuning. It is not the paper's own dataset; it is a stand-in for agricultural text.

## Models
- **Baseline:** longest-match dictionary lookup, treating the training entity strings as an expert ontology list. (This is not the paper's LLM method.)
- **Improved:** scikit-learn logistic-regression BIO token tagger with word, suffix, shape, neighbouring-word and ontology-match features. The regularisation strength C is tuned on the held-out part of train.

Both are scored on the same test split with the same hand-written exact-match entity-level F1 (precision and recall are secondary metrics).

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> S Wilson, A Ginige, J Goonatilake (2024). *Design science research approach for ontology development in agriculture: utilising advances of LLM for automated entity extraction*. aisel.aisnet.org. https://aisel.aisnet.org/acis2024/150/

Paper author: Prof. Athula Ginige, Western Sydney University (a.ginige@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
