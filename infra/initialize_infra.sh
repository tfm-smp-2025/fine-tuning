#!/bin/sh

set -eu

cd "$(dirname "$0")"

docker-compose up -d

if which jq >>/dev/null 2>>/dev/null;then
    # Load models from config
    MODELS=$(
        cat config \
        | tr  '\n' ' ' \
        | sed -r  's/.*MODELS=`(.*)`.*/\1/' \
        | jq .[].endpoints[].ollamaName -r
    )

    echo "$MODELS" > models.txt
else
    echo "Loading models from /cached/ list"
    MODELS=$(cat models.txt)
fi

set -x
for model in $MODELS;do
    docker-compose exec ollama-service ollama pull "$model"
done