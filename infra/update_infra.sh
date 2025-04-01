#!/bin/sh

set -eux

cd "$(dirname "$0")"

docker pull ollama/ollama
docker pull ghcr.io/huggingface/chat-ui-db:latest

docker-compose down
docker-compose up -d
