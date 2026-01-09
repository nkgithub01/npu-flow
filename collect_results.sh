#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PNR_COMMIT_HASH=$(git rev-parse --short HEAD:./npu-pnr)
OUTPUT_PATH="./Results"
PNR_OUTPUT_PATH="${OUTPUT_PATH}/PnR_commit_${PNR_COMMIT_HASH}"

# Collect hand placed results
RUN_NAME="Hand_Placed"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type "hand_placed" \
    --router.type "aie" \
    --run \
    --output-dir "${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect sequential placer results
RUN_NAME="Sequential_Placed"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type "aie" \
    --router.type "aie" \
    --run \
    --output-dir "${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect SAPlacer results with MILP Costing Estimation
RUN_NAME="SAPlacer_MILP_Cost"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-6 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 5 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 100 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.enable_initial_placement_randomization \
        --placer.cost_estimator milp \
        --logger.verbose minimal \
        --timeout_sec 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Try AIE router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AIE_Router"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect SAPlacer results with Only Cost Estimation
RUN_NAME="SAPlacer_Cost_Estimation"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-6 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 5 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.enable_initial_placement_randomization \
        --placer.cost_estimator bb \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Try AIE router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AIE_Router"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect SAPlacer results with Cost and Congestion Estimation
RUN_NAME="SAPlacer_Cost_Congestion_Estimation"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-6 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 5 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.enable_initial_placement_randomization \
        --placer.cost_estimator prob \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Try AIE router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AIE_Router"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect One shot MILP results
RUN_NAME="MILPPlacer"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --placer.pnr.type milp \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --logger.verbose minimal \
        --timeout_sec 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Try AIE router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AIE_Router"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"

# Collect LSMO results
RUN_NAME="LSMOPlacer"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --placer.pnr.type ls \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --logger.verbose minimal \
        --timeout_sec 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Try AIE router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AIE_Router"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"