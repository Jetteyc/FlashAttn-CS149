#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [0|1|2]..."
    echo "  0: run simple test -> output/test/0-1.txt"
    echo "  1: run FA2 forward benchmark -> output/fa1/forward.{txt,json}"
    echo "  2: run FA2 backward benchmark -> output/fa1/backward.{txt,json}"
    echo
    echo "Optional env vars:"
    echo "  FA1_N=2048                    # sequence length for part 1"
    echo "  FA1_D=64                      # head dimension for part 1"
    echo "  FA1_BC=64                     # KV tile length along seqlen (Bc)"
    echo "  FA1_BR=32                     # Q tile (Br)"
    echo "  FA1_CAUSAL=1                  # pass --causal to fa2 / fa2_bw"
    echo "  FA1_BW_N=1024                 # sequence length for part 2"
    echo "  FA1_BW_D=64                   # head dimension for part 2"
    echo "  FA1_PROFILE_NCU=1             # enable Nsight Compute capture (.ncu-rep)"
    echo "  FA1_PROFILE_ITERS=1           # pass --profile-iters"
    echo "  FA1_NCU_BIN=ncu               # Nsight Compute CLI binary"
    echo "  PYTHON_BIN=python             # python executable name"
}

mkdir -p output/test output/fa1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.0+PTX}"
export NVTE_CUDA_ARCHITECTURES="${NVTE_CUDA_ARCHITECTURES:-120}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FA1_NCU_BIN="${FA1_NCU_BIN:-ncu}"

FA1_N="${FA1_N:-2048}"
FA1_D="${FA1_D:-64}"
FA1_BC="${FA1_BC:-64}"
FA1_BR="${FA1_BR:-32}"
FA1_CAUSAL="${FA1_CAUSAL:-0}"
FA1_PROFILE_NCU="${FA1_PROFILE_NCU:-0}"
FA1_PROFILE_ITERS="${FA1_PROFILE_ITERS:-1}"

_extra_causal() {
    if [ "${FA1_CAUSAL}" = "1" ]; then
        echo "--causal"
    fi
}

_extra_profile() {
    if [ "${FA1_PROFILE_NCU}" = "1" ]; then
        echo "--profile-ncu --profile-iters ${FA1_PROFILE_ITERS}"
    fi
}

_run_py() {
    "${PYTHON_BIN}" "$@"
}

_run_py_ncu() {
    local rep_base="$1"
    shift
    "${FA1_NCU_BIN}" \
        --profile-from-start off \
        --nvtx \
        --export "${rep_base}" \
        "${PYTHON_BIN}" "$@"
}

run_part0() {
    echo "[run.sh] Running part 0 (simple test)"
    _run_py gpt149.py test > output/test/0-1.txt
    echo "[run.sh] Wrote output/test/0-1.txt"
}

run_part1() {
    local out_file="output/fa1/forward.txt"
    local json_file="output/fa1/forward.json"
    local rep_base="output/fa1/forward"
    echo "[run.sh] Running part 1 (FA2 forward), bc=${FA1_BC}, br=${FA1_BR}, N=${FA1_N}, d=${FA1_D}"
    if [ "${FA1_PROFILE_NCU}" = "1" ]; then
        # shellcheck disable=SC2046
        _run_py_ncu "${rep_base}" gpt149.py fa2 -N "${FA1_N}" -d "${FA1_D}" -bc "${FA1_BC}" -br "${FA1_BR}" --json-out "${json_file}" $(_extra_causal) $(_extra_profile) > "${out_file}" 2>&1
        echo "[run.sh] Wrote ${rep_base}.ncu-rep"
    else
        # shellcheck disable=SC2046
        _run_py gpt149.py fa2 -N "${FA1_N}" -d "${FA1_D}" -bc "${FA1_BC}" -br "${FA1_BR}" --json-out "${json_file}" $(_extra_causal) $(_extra_profile) > "${out_file}" 2>&1
    fi
    echo "[run.sh] Wrote ${out_file}"
    echo "[run.sh] Wrote ${json_file}"
}

run_part2() {
    local n="${FA1_BW_N:-1024}"
    local d="${FA1_BW_D:-64}"
    local json_file="output/fa1/backward.json"
    local rep_base="output/fa1/backward"
    echo "[run.sh] Running part 2 (FA2 backward benchmark), bc=${FA1_BC}, br=${FA1_BR}, N=${n}, d=${d}"
    if [ "${FA1_PROFILE_NCU}" = "1" ]; then
        # shellcheck disable=SC2046
        _run_py_ncu "${rep_base}" gpt149.py fa2_bw -N "${n}" -d "${d}" -bc "${FA1_BC}" -br "${FA1_BR}" --json-out "${json_file}" $(_extra_causal) $(_extra_profile) > output/fa1/backward.txt 2>&1
        echo "[run.sh] Wrote ${rep_base}.ncu-rep"
    else
        # shellcheck disable=SC2046
        _run_py gpt149.py fa2_bw -N "${n}" -d "${d}" -bc "${FA1_BC}" -br "${FA1_BR}" --json-out "${json_file}" $(_extra_causal) $(_extra_profile) > output/fa1/backward.txt 2>&1
    fi
    echo "[run.sh] Wrote output/fa1/backward.txt"
    echo "[run.sh] Wrote ${json_file}"
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
