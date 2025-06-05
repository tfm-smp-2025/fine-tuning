# fine-tuning

This repository holds the code to fine-tune LLMs on mapping natural language queries into SPARQL queries.

## Preparation

### Pull data

Run `python3 scripts/pull_datasets.py` (no dependencies).

#### Datasets used

The following datasets are used for this fine tuning. Most of them provide separate sets for training and testing.

- **qald-9**: https://github.com/ag-sc/QALD/
- **qald-10**: https://github.com/KGQA/QALD-10/
- **beastiary**: https://github.com/danrd/sparqlgen/
- **VQuAnDA**: https://github.com/AskNowQA/VQUANDA/
- **LC-QuAD 1.0**: https://github.com/AskNowQA/LC-QuAD/
- **LC-QuAD 2.0**: https://github.com/AskNowQA/LC-QuAD2.0/
- **DBNQA**: https://github.com/AKSW/DBNQA

### Install dependencies

Run `pip install -r requirements.txt`.

### (Recommended) Remote infrastructure

Consider using a remote infrastructure as describe on the [unattended-cloud-fine-tuning](https://github.com/tfm-smp-2025/unattended-cloud-fine-tuning) repo.

### (Deprecated) Local Infrastructure

The simplest way to instantiate the necessary infrastructure for these tests is to use [Docker Compose](https://docs.docker.com/compose/). After installing it run `scripts/initialize_infra.sh`, this will run two steps:

1. Run `docker compose up -d` to start the servers.
2. When the startup is complete, it pulls the models configured on `infra/config` (if `jq` is installed, if not the list is pulled from `models.txt`).

When you're done with the tests, you can stop the servers with `docker compose down` on the `infra` directory too.

To follow Ollama's logs, run `docker-compose logs -f` on the `infra` directory.

## Running

After the dependencies are installed, the [unattended-cloud-fine-tuning/notebooks directory](https://github.com/tfm-smp-2025/unattended-cloud-fine-tuning/tree/main/notebooks) has examples on how this tool can be invoked to test a given model. For example, to test [Microsoft's Phi-4](https://huggingface.co/microsoft/phi-4) model [we can run this command](https://github.com/tfm-smp-2025/unattended-cloud-fine-tuning/blob/main/notebooks/evaluate-model-phi4-14b.ipynb):

```bash
WEAVIATE_HOST=<URL_TO_SERVER_HOLDING_THE_VECTOR_DB> \
    python3 \
    -m src \
    --seed 42 \
    test --models="phi-4:14b" \
    --sparql-server '<URL_TO_SERVER_HOLDING_THE_KNOWLEDGE_BASE>' \
    --sample 100 \
    --dataset 'LC-QuAD 1.0'
```