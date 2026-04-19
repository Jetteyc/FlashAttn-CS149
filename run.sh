#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [0|1|2]..."
    echo "  0: run simple test -> output/test/0-1.txt"
    echo "  1: run one fa1 forward benchmark (default N=2048, d=64)"
    echo "  2: run fa1 backward smoke test -> output/fa1/backward.txt"
    echo
    echo "Optional env vars:"
    echo "  FA1_N=2048                    # sequence length for part 1"
    echo "  FA1_D=64                      # head dimension for part 1"
    echo "  FA1_BC=64                     # block column size"
    echo "  FA1_BR=64                     # block row size"
    echo "  FA1_IMPL=cuda                 # student kernel: cuda or wmma"
}

mkdir -p output/test output/fa1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.0+PTX}"
export NVTE_CUDA_ARCHITECTURES="${NVTE_CUDA_ARCHITECTURES:-120}"

FA1_N="${FA1_N:-2048}"
FA1_D="${FA1_D:-64}"
FA1_BC="${FA1_BC:-64}"
FA1_BR="${FA1_BR:-32}"
FA1_IMPL="${FA1_IMPL:-cuda}"

run_part0() {
    echo "[run.sh] Running part 0 (simple test)"
    python3 gpt149.py test > output/test/0-1.txt
    echo "[run.sh] Wrote output/test/0-1.txt"
}

run_part1() {
    local out_file="output/fa1/forward.txt"
    echo "[run.sh] Running part 1 (fa1), bc=${FA1_BC}, br=${FA1_BR}, N=${FA1_N}, d=${FA1_D}, impl=${FA1_IMPL}"
    python3 gpt149.py fa1 -N "${FA1_N}" -d "${FA1_D}" -bc "${FA1_BC}" -br "${FA1_BR}" --impl "${FA1_IMPL}" > "${out_file}"
    echo "[run.sh] Wrote ${out_file}"
}

run_part2() {
    local n="${FA1_BW_N:-128}"
    local d="${FA1_BW_D:-64}"
    echo "[run.sh] Running part 2 (fa1 backward smoke), bc=${FA1_BC}, br=${FA1_BR}, N=${n}, d=${d}, impl=${FA1_IMPL}"
    python3 gpt149.py fa1_bw -N "${n}" -d "${d}" -bc "${FA1_BC}" -br "${FA1_BR}" --impl "${FA1_IMPL}" > output/fa1/backward.txt
    echo "[run.sh] Wrote output/fa1/backward.txt"
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