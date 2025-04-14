# fine-tuning

This repository holds the code to fine-tune LLMs on mapping natural language queries into SPARQL queries.

## Preparation

### Pull data

Run `python3 scripts/pull_datasets.py` (no dependencies).

### Infrastructure

The simplest way to instantiate the necessary infrastructure for these tests is to use [Docker Compose](https://docs.docker.com/compose/). After installing it run `scripts/initialize_infra.sh`, this will run two steps:

1. Run `docker compose up -d` to start the servers.
2. When the startup is complete, it pulls the models configured on `infra/config` (if `jq` is installed, if not the list is pulled from `models.txt`).

When you're done with the tests, you can stop the servers with `docker compose down` on the `infra` directory too.

To follow Ollama's logs, run `docker-compose logs -f` on the `infra` directory.

## References

### Datasets used

The following datasets are used for this fine tuning. Most of them provide separate sets for training and testing.

- **qald-9**: https://github.com/ag-sc/QALD/
- **qald-10**: https://github.com/KGQA/QALD-10/
- **beastiary**: https://github.com/danrd/sparqlgen/
- **VQuAnDA**: https://github.com/AskNowQA/VQUANDA/
- **LC-QuAD 1.0**: https://github.com/AskNowQA/LC-QuAD/
- **LC-QuAD 2.0**: https://github.com/AskNowQA/LC-QuAD2.0/
- **DBNQA**: https://github.com/AKSW/DBNQA
