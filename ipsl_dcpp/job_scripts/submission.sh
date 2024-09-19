#!/usr/bin/env bash
#see https://sharats.me/posts/shell-script-best-practices/ for explanation on many things in this script
set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi
gpu_version=$1
experiment_name=$2



if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'Usage: ./submission.sh ${a100|v100} $experiment_name'
    exit
fi

cd $NEWSCRATCH

main() {
    #need to load 
    cd $NEWSCRATCH
    if [[ "$gpu_version" == "a100" ]]; then
        module load cpuarch/amd
    fi
    module load pytorch-gpu/py3/2.2.0

    mkdir $experiment_name
    cd $experiment_name
    git clone "https://github.com/grahamclyne/ipsl_dcpp.git"
    cd ipsl_dcpp/ipsl_dcpp
    if [[ "$gpu_version" == "a100" ]]; then 
        python submit.py cluster="jean_zay_a100"
    else
        python submit.py cluster="jean_zay_v100" batch_size=1
    fi
};


main "$@"