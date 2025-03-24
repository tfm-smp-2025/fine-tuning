# fine-tuning

This repository holds the code to fine-tune LLMs on mapping natural language queries into SPARQL queries.

## Preparation

### Pull data

Run `python3 scripts/pull_datasets.py` (no dependencies).

## References

### Datasets used

The following datasets are used for this fine tuning. Most of them provide separate sets for training and testing.

- **qald-9**: https://github.com/ag-sc/QALD/
- **qald-10**: https://github.com/KGQA/QALD-10/
- **bestiary**: https://github.com/danrd/sparqlgen/
- **VQuAnDA**: https://github.com/AskNowQA/VQUANDA/
- **LC-QuAD 1.0**: https://github.com/AskNowQA/LC-QuAD/
- **LC-QuAD 2.0**: https://github.com/AskNowQA/LC-QuAD2.0/
- **DBNQA**: https://github.com/AKSW/DBNQA
