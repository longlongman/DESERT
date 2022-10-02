#!/bin/bash


function parse_args(){
    while [[ "$#" -gt 0 ]]; do
        found=0
        for key in "${!BASH_ARGS[@]}"; do
            if [[ "--$key" == "$1" ]] ; then
                BASH_ARGS[$key]=$2
                found=1
            fi
        done
        if [[ $found == 0 ]]; then
            echo "arg $1 not defined!" >&2
            exit 1
        fi
        shift; shift
    done

    echo "======== PARSED BASH ARGS ========" >&2
    for key in "${!BASH_ARGS[@]}"; do
        echo "    $key = ${BASH_ARGS[$key]}" >&2
        eval "$key=${BASH_ARGS[$key]}" >&2
    done
    echo "==================================" >&2
}
