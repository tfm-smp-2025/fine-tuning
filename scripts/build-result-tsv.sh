#!/usr/bin/env bash

set -eu

cd "$(dirname "$0")"

SRC_DIRECTORY=${1:-experiment-viewer/logs/}
REMOTE_HOST=${REMOTE_HOST:-}

if [ -z "${REMOTE_HOST}" ];then
    WORK_DIR="${SRC_DIRECTORY}"
else
    WORK_DIR="$(mktemp --directory --suffix="-tfm-smp-2025-build-csv")"
    rsync -HPaz "${REMOTE_HOST}:${SRC_DIRECTORY}" "${WORK_DIR}" >&2
    TO_REMOVE_DIR="${WORK_DIR}"
fi

echo -e 'File\tModel\tStart time\tEnd time\tTime diff (seconds)\tsame_value_and_type\tsame_value\tsame_length\tunhandled_test_check\tno_match\terror'

# same_value_and_type
# same_value
# same_length
# unhandled_test_check
# no_match
# error

for fname in "${WORK_DIR}"/*;
do
    echo "$fname" >&2
    levels=$(cat "$fname"| jq -r 'select(.operation=="test_result")|.data.result' \
        | sort \
        | uniq -c
    )
    echo -ne "$(basename "$fname")\t"

    cat "$fname"| jq -r .data.parameters.translator.model_name 2>/dev/null |grep -v null|head -n1|tr -d '\n'
    echo -ne "\t" # Model
    
    head -n1 "$fname" |jq -r '.time'|tr -d '\n'
    echo -ne "\t" # Start time

    tail -n1 "$fname" |jq -r '.time'|tr -d '\n'
    echo -ne "\t" # End time time (tab added by next echo)

    # Timediff (tab added by next echo)
    tstart=$(head -n1 "$fname" |jq -r '.timestamp'|tr -d '\n')
    tend=$(tail -n1 "$fname" |jq -r '.timestamp'|tr -d '\n')
    python3 -c "print('{:.3f}'.format($tend - $tstart), end='')"

    for lv in same_value_and_type same_value same_length unhandled_test_check no_match error;
    do
        value=$(echo "$levels" |grep -P "\s*\d+\s+$lv"|awk '{ print $1;}'|tr -d '\n')
        if [[ $(echo "$value"|grep .|wc -l) -eq 0 ]];then
            value=0
        fi

        echo -ne "\t$value"
    done
    echo
done

# if [ ! -z "${TO_REMOVE_DIR}" ];then
#     rm -Rf "${TO_REMOVE_DIR}"
# fi