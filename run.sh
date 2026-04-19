#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [0|1|2]..."
    echo "  0: run simple test -> output/test/0-1.txt"
    echo "  1: run fa1 benchmarks (default N list: 128, 512, 2048)"
    echo "  2: run fa1 backward smoke test -> output/fa1/2-1.txt"
    echo
    echo "Optional env vars:"
    echo "  FA1_N_LIST=\"128 512 2048\"   # sequence lengths for part 1"
    echo "  FA1_BC=32                     # block column size"
    echo "  FA1_BR=32                     # block row size"
}

mkdir -p output/test output/fa1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.0+PTX}"
export NVTE_CUDA_ARCHITECTURES="${NVTE_CUDA_ARCHITECTURES:-120}"

FA1_N_LIST="${FA1_N_LIST:-128 512 2048}"
FA1_BC="${FA1_BC:-32}"
FA1_BR="${FA1_BR:-32}"

run_part0() {
    echo "[run.sh] Running part 0 (simple test)"
    python3 gpt149.py test > output/test/0-1.txt
    echo "[run.sh] Wrote output/test/0-1.txt"
}

run_part1() {
    echo "[run.sh] Running part 1 (fa1), bc=${FA1_BC}, br=${FA1_BR}, N list: ${FA1_N_LIST}"
    local idx=1
    for n in ${FA1_N_LIST}; do
        local out_file="output/fa1/1-${idx}.txt"
        echo "[run.sh]   -> N=${n}, output=${out_file}"
        python3 gpt149.py fa1 -N "${n}" -bc "${FA1_BC}" -br "${FA1_BR}" > "${out_file}"
        idx=$((idx + 1))
    done
}

run_part2() {
    local n="${FA1_BW_N:-128}"
    echo "[run.sh] Running part 2 (fa1 backward smoke), bc=${FA1_BC}, br=${FA1_BR}, N=${n}"
    python3 gpt149.py fa1_bw -N "${n}" -bc "${FA1_BC}" -br "${FA1_BR}" > output/fa1/2-1.txt
    echo "[run.sh] Wrote output/fa1/2-1.txt"
}

if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi

for arg in "$@"; do
    case "${arg}" in
        0)
            run_part0
            ;;
        1)
            run_part1
            ;;
        2)
            run_part2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown part number: ${arg}"
            usage
            exit 1
            ;;
    esac
done