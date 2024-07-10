#!/usr/bin/env bash
#see https://sharats.me/posts/shell-script-best-practices/ for explanation on many things in this script
set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi
gpu_version=$1




if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'Usage: ./submission.sh ${a100|v100}'
    exit
fi

cd $SCRATCH

main() {
    module load pytorch-gpu/py3/2.2.0
    #need to load 
    cd $SCRATCH
    if [[ "$gpu_version" == "a100" ]]; then
        module load cpuarch/amd
    fi
    ls
};

main "$@"