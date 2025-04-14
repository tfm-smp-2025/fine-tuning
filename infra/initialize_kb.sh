#!/bin/sh

set -eu

DOCKER_CONTAINER_NAME='apache_jena_fuseki'
ADMIN_SPARQL_PASSWORD="$ADMIN_SPARQL_PASSWORD"
SPARQL_HOST="http://127.0.0.1:3030"

cd "$(dirname "$0")"

cd ../datasets/

get_password() {
    echo -n "admin:$ADMIN_SPARQL_PASSWORD" |base64|tr -d '\n'
}

create_dataset() {
    curl -f "$SPARQL_HOST"'/$/datasets' \
         -X POST \
         -H 'Content-Type: application/x-www-form-urlencoded' \
         -H "Origin: $SPARQL_HOST" \
         -H "Authorization: Basic $(get_password)" \
         --data-raw "dbName=$1&dbType=tdb2"
}

delete_dataset() {
    curl -f "$SPARQL_HOST"'/$/datasets/'"$1" \
         -X DELETE \
         -H "Origin: $SPARQL_HOST" \
         -H "Authorization: Basic $(get_password)"
}

prepare_file() {
    fname="$1"

    # Check if we have `pv` for progress view
    if which -s pv ;then
        PV=pv
    else 
        PV=cat
    fi

    name=$(basename "$fname"|sed -r 's/\.[^.]+$//')

    if echo "$fname" | grep -q '.bz2$';then
        echo "Unpacking..." >&2
        tmpname=$(mktemp -t "tmp_XXXXXXXXXX_$name")
        $PV "$fname" | bzcat > "$tmpname"
        echo "$tmpname"
    elif echo "$fname" | grep -q '.gz$';then
        echo "Unpacking..." >&2
        tmpname=$(mktemp -t "tmp_XXXXXXXXXX_$name")
        $PV "$fname" | gunzip > "$tmpname"
        echo "$tmpname"
    else
        echo "$fname"
    fi
}

load_file_in_dataset() {
    dataset="$1"
    file="$2"
    size=$(du -h "$2"|cut -d\  -f1)
    echo "Loading '$file' into '$dataset' ($size)"

    # Temporarily extract file if appropriate
    unpacked_file=$(prepare_file "$file")

    curl -f "$SPARQL_HOST/$dataset/data" \
        --compressed \
        -X POST \
        -H 'Content-Type: multipart/form-data' \
        -H "Origin: $SPARQL_HOST" \
        -H "Authorization: Basic $(get_password)" \
        -F "file=@$unpacked_file"

    # Cleanup
    if [ "$unpacked_file" != "$file" ];then
        # Assert that it's on /tmp/
        echo "$unpacked_file"| grep '^/tmp/'
        rm -f "$unpacked_file"
    fi
}

load_dir_in_dataset() {
    dataset="$1"
    dir="$2"
    for file in $dir/*;do
        load_file_in_dataset "$dataset" "$file"
    done
}

# Beastiary dataset
printf '\e[1m== Loading beastiary...\e[0m\n'

# Allow for errors when deleting this dataset, just in 
#  case it doesn't exist previsouly.
set +e
delete_dataset beastiary
set -e
create_dataset beastiary
load_file_in_dataset "beastiary" "by_url/github.com/danrd/sparqlgen/raw/refs/heads/main/beastiary_kg.rdf"

printf '\e[1m== Loading DBPedia 2016v04\e[0m\n'
set +e
delete_dataset dbpedia_2016_04
set -e

create_dataset dbpedia_2016_04
load_dir_in_dataset "dbpedia_2016_04" "by_url/downloads.dbpedia.org/2016-04/core"

create_dataset dbpedia_2016_10
for lang in by_url/downloads.dbpedia.org/2016-10/core-i18n/*;do
    printf '\e[1m== Loading DBPedia 2016-10\e[0m\n'
    load_dir_in_dataset "dbpedia_2016_10" "$lang"
done
