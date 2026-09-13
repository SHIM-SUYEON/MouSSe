#!/usr/bin/env bash
# eval.sh
#
# Evaluation driver for the MUSSE PIT+UQ speech-separation model.
# Experiments run in parallel pairs on 2 GPUs (GPU_A, GPU_B).
# Run from the repository root, e.g.  bash inference/eval.sh main
#
# Usage:
# bash inference/eval.sh            # everything (main + adaptive sweep + summary)
# bash inference/eval.sh main       # backbone + adaptive (cap=1300, paper numbers)
# bash inference/eval.sh adaptive   # adaptive lambda-scale sweep
# bash inference/eval.sh summary    # print summary tables from existing logs

set -euo pipefail

# Paths 
SCRIPT="$(dirname "$0")/eval.py"
CKPT="/data/MOUSSE/mousse-checkpoint.pt"
OUT_BASE="results/mousse_uq"
LOG_DIR="logs/mousse_uq_1300"

# Config 
GPU_A="0"      # first GPU
GPU_B="1"      # second GPU
CAP_MAIN=1300  # full test set for final reported numbers
CAP_SWEEP=1300 # adaptive lambda-scale sweep

mkdir -p "$LOG_DIR"

# Helpers 
_run() {
    # _run <log> <extra args...>
    local log="$1"; shift
    echo "[RUN] $* → $log"
    python3 -u "$SCRIPT" "$@" > "$log" 2>&1
}

run_adaptive() {
    local scale="$1" gpu="$2" cap="${3:-$CAP_SWEEP}"
    local tag; tag=$(echo "$scale" | tr '.' '_')
    local log="$LOG_DIR/adaptive_scale_${tag}.log"
    local out="$OUT_BASE/adaptive_scale_${tag}"
    _run "$log" --ckpt "$CKPT" \
        --lambda-scale "$scale" --gpu "$gpu" --cap "$cap" --out "$out"
}

# Section 0: Main comparison (full cap=1300 for paper numbers) 
run_main() {
    echo ""
    echo "======================================================================"
    echo " MAIN  (CAP=$CAP_MAIN — for paper numbers)"
    echo "======================================================================"

    # Backbone baseline (no UQ guidance, fixed alpha=1.5)
    LOG_BB="$LOG_DIR/main_backbone.log"
    _run "$LOG_BB" \
        --ckpt "$CKPT" --ablation no_uq \
        --gpu "$GPU_A" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_backbone" &
    PID_A=$!

    # Adaptive UQ guidance (lambda_scale=1.0 → lambda* ≈ 0.50)
    LOG_AD="$LOG_DIR/main_adaptive_1_0.log"
    _run "$LOG_AD" \
        --ckpt "$CKPT" --lambda-scale 1.0 \
        --gpu "$GPU_B" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_adaptive_1_0" &
    PID_B=$!

    wait $PID_A $PID_B
    echo "[main] backbone + adaptive_1.0 done."
}

# Section 1: Adaptive-lambda (lambda_scale) sweep 
run_adaptive_sweep() {
    echo ""
    echo "======================================================================"
    echo " ADAPTIVE LAMBDA_SCALE SWEEP  (CAP=$CAP_SWEEP)"
    echo "======================================================================"

    run_adaptive 0.0  "$GPU_A" "$CAP_SWEEP" &
    wait

    run_adaptive 0.5  "$GPU_A" "$CAP_SWEEP" &
    run_adaptive 1.0  "$GPU_B" "$CAP_SWEEP" &
    wait

    run_adaptive 1.5  "$GPU_A" "$CAP_SWEEP" &
    run_adaptive 2.0  "$GPU_B" "$CAP_SWEEP" &
    wait

    echo "[sweep] Adaptive lambda_scale sweep complete."
}

# Section 2: Summary 
print_summary() {
    echo ""
    echo "======================================================================"
    echo " SUMMARY  (SDRi, SI-SDRi, ESTOI, ESTOIi from [OVERALL] lines)"
    echo "======================================================================"

    echo ""
    echo "--- Main comparison (cap=$CAP_MAIN) ---"
    for tag in main_backbone main_adaptive_1_5; do
        local log="$LOG_DIR/${tag}.log"
        if [[ -f "$log" ]]; then
            overall=$(grep "^\[OVERALL\]" "$log" | tail -1 || echo "(no OVERALL line yet)")
            echo "  [$tag] $overall"
        else
            echo "  [$tag] (not run)"
        fi
    done

    echo ""
    echo "--- Adaptive lambda_scale sweep ---"
    for scale in 0.0 0.5 1.0 1.5 2.0; do
        local tag; tag=$(echo "$scale" | tr '.' '_')
        local log="$LOG_DIR/adaptive_scale_${tag}.log"
        if [[ -f "$log" ]]; then
            overall=$(grep "^\[OVERALL\]" "$log" | tail -1 || echo "(no OVERALL line yet)")
            printf "  scale=%-5s %s\n" "$scale" "$overall"
        else
            printf "  scale=%-5s (not run)\n" "$scale"
        fi
    done
}

# Entry point 
MODE="${1:-all}"

case "$MODE" in
    main)
        run_main
        print_summary
        ;;
    adaptive)
        run_adaptive_sweep
        print_summary
        ;;
    summary)
        print_summary
        ;;
    all)
        run_main
        run_adaptive_sweep
        print_summary
        ;;
    *)
        echo "Usage: bash $0 [main|adaptive|summary|all]"
        exit 1
        ;;
esac
