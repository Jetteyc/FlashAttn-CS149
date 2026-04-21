#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [0|1|2]..."
    echo "  0: run simple test -> output/test/0-1.txt"
    echo "  1: run FA2 forward benchmark (default N=2048, d=64)"
    echo "  2: run FA2 backward  -> output/fa1/backward.txt"
    echo
    echo "Optional env vars:"
    echo "  FA1_N=2048                    # sequence length for part 1"
    echo "  FA1_D=64                      # head dimension for part 1"
    echo "  FA1_BC=64                     # KV tile length along seqlen (Bc)"
    echo "  FA1_BR=32                     # Q tile (Br)"
    echo "  FA1_CAUSAL=1                 # pass --causal to fa2 / fa2_bw"
    echo "  CONDA_ENV=megatron-lm-autotuner  # conda env used by run.sh"
    echo "  PYTHON_BIN=python            # python executable name"
}

mkdir -p output/test output/fa1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.0+PTX}"
export NVTE_CUDA_ARCHITECTURES="${NVTE_CUDA_ARCHITECTURES:-120}"
CONDA_ENV="${CONDA_ENV:-megatron-lm-autotuner}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FA1_N="${FA1_N:-2048}"
FA1_D="${FA1_D:-64}"
FA1_BC="${FA1_BC:-64}"
FA1_BR="${FA1_BR:-32}"
FA1_CAUSAL="${FA1_CAUSAL:-0}"

_extra_causal() {
    if [ "${FA1_CAUSAL}" = "1" ]; then
        echo "--causal"
    fi
}

_run_py() {
    if command -v conda >/dev/null 2>&1; then
        conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" "$@"
    else
        "${PYTHON_BIN}" "$@"
    fi
}

run_part0() {
    echo "[run.sh] Running part 0 (simple test)"
    _run_py gpt149.py test > output/test/0-1.txt
    echo "[run.sh] Wrote output/test/0-1.txt"
}

run_part1() {
    local out_file="output/fa1/forward.txt"
    echo "[run.sh] Running part 1 (FA2 forward), env=${CONDA_ENV}, bc=${FA1_BC}, br=${FA1_BR}, N=${FA1_N}, d=${FA1_D}"
    # shellcheck disable=SC2046
    _run_py gpt149.py fa2 -N "${FA1_N}" -d "${FA1_D}" -bc "${FA1_BC}" -br "${FA1_BR}" $(_extra_causal) > "${out_file}"
    echo "[run.sh] Wrote ${out_file}"
}

run_part2() {
    local n="${FA1_BW_N:-128}"
    local d="${FA1_BW_D:-64}"
    echo "[run.sh] Running part 2 (FA2 backward), env=${CONDA_ENV}, bc=${FA1_BC}, br=${FA1_BR}, N=${n}, d=${d}"
    # shellcheck disable=SC2046
    _run_py gpt149.py fa2_bw -N "${n}" -d "${d}" -bc "${FA1_BC}" -br "${FA1_BR}" $(_extra_causal) > output/fa1/backward.txt
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
