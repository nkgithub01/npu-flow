#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PNR_COMMIT_HASH=$(git rev-parse --short HEAD:./npu-pnr)
OUTPUT_PATH="./Results"
OUTPUT_PATH="${OUTPUT_PATH}/PnR_commit_${PNR_COMMIT_HASH}"
tasklist="benchmarks/tasklist.yml"
num_runs=5
declare -A results_csv_list
declare -A run_name_list
declare -a results_order

# Collect hand placed results
RUN_NAME="Hand_Placed"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type hand_placed \
    --router.type aie \
    --run \
    --output-dir "${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="HP"
    results_csv_list["${RUN_NAME}"]="${PNR_OUTPUT_DIR}/results.csv"
fi
# Try PNR MILP router on the above results
RUN_NAME="Hand_Placed_PNR_MILP_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type hand_placed \
    --router.type pnr \
    --router.pnr.type milp \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="HP_PNR"
    results_csv_list["${RUN_NAME}"]="${PNR_OUTPUT_DIR}/results.csv"
fi
# Try PNR PathFinder router on the above results
# NOTE: PathFinder needs more RAM, so use fewer parallel jobs
RUN_NAME="Hand_Placed_PNR_PathFinder_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type hand_placed \
    --router.type pnr \
    --router.pnr.type pf \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 2
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="HP_PF"
    results_csv_list["${RUN_NAME}"]="${PNR_OUTPUT_DIR}/results.csv"
fi
# Collect sequential placer results
RUN_NAME="Sequential_Placer"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type aie \
    --router.type aie \
    --run \
    --output-dir "${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SP"
    results_csv_list["${RUN_NAME}"]="${PNR_OUTPUT_DIR}/results.csv"
fi

for seed in $(seq 1 $num_runs); do
# Collect SAPlacer results with Only Bounding Box Cost Estimation
RUN_NAME="SAPlacer_Bounding_Box_Cost_Estimation"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-1 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 2 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.initial_temperature_multiplier 1 \
        --placer.cost_estimator bb \
        --placer.enable_cost_legality_estimator \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"

# Collect SAPlacer results with Cost and Congestion Estimation
RUN_NAME="SAPlacer_Cost_Congestion_Estimation_channel_uniform"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-1 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 2 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.initial_temperature_multiplier 1 \
        --placer.cost_estimator prob \
        --placer.cost_estimator_probability_distribution channel_uniform \
        --placer.enable_cost_legality_estimator \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_BBCG"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_BBCG_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"

RUN_NAME="SAPlacer_Cost_Congestion_Estimation_path_uniform"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-1 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 2 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.initial_temperature_multiplier 1 \
        --placer.cost_estimator prob \
        --placer.cost_estimator_probability_distribution path_uniform \
        --placer.enable_cost_legality_estimator \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_BBCG_path_uniform"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_BBCG_path_uniform_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"

# Collect SAPlacer results with MILP Costing Estimation
RUN_NAME="SAPlacer_MILP_Cost"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type sa \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.greedy_stage_entering_temperature 1e-1 \
        --placer.greedy_stage_entering_acceptance_ratio 0.001 \
        --placer.greedy_stage_num_iters_scaling_factor 2 \
        --placer.max_iters 1000000 \
        --placer.greedy_stage_max_iters 10000 \
        --placer.max_move_attempts 2000000000 \
        --placer.num_moves_per_iter 10000 \
        --placer.enable_dynamic_temperature_scheduling \
        --placer.initial_temperature_multiplier 1 \
        --placer.cost_estimator milp \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_MILP"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="SA_MILP_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"

# Collect One shot MILP results
RUN_NAME="MILPPlacer"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type milp \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="MILP"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="MILP_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"

# Collect LSMO results
RUN_NAME="LSMOPlacer"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --placer.pnr.type ls \
    --router.type pnr \
    --router.pnr.type milp \
    --placer.pnr.args=" \
        --placer.enable_aggressive_local_search \
        --placer.neighbor_region_shape cross \
        --placer.max_consecutive_iters_no_best_cost_improvement_scaling_factor 0.1 \
        --placer.random_seed ${seed} \
        --logger.verbose minimal \
        --timeout_secs 3600 \
    " \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="LSMO"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
# Try AMD router on the above results
IMPORT_DIR=$PNR_OUTPUT_DIR
RUN_NAME="${RUN_NAME}_AMD_Router"
PNR_OUTPUT_DIR="${OUTPUT_PATH}/${RUN_NAME}_seed_${seed}_${TIMESTAMP}"
python3 utils/build_benchmarks.py "${tasklist}" \
    --placer.type pnr \
    --router.type aie \
    --import-pnr-results "${IMPORT_DIR}" \
    --import-pnr-iter -1 \
    --run \
    --output-dir="${PNR_OUTPUT_DIR}" \
    -j 20
python utils/parse_results.py \
    --tasklists "${tasklist}" \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/results.csv"
if [[ ! -v run_name_list["${RUN_NAME}"] ]]; then
    results_order+=("${RUN_NAME}")
    run_name_list["${RUN_NAME}"]="LSMO_AMD"
fi
results_csv_list["${RUN_NAME}"]+=" ${PNR_OUTPUT_DIR}/results.csv"
done

# After collecting all results, average the runs with different seeds
for run_name in "${!run_name_list[@]}"; do
    echo "Averaging results for ${run_name}..."
    csv_files=${results_csv_list[$run_name]}
    python utils/average_results.py \
        -f ${csv_files} \
        -o "${OUTPUT_PATH}/${run_name}_averaged_results_${TIMESTAMP}.csv"
done

# Organize the final results in a Excel file in a predefined order
results_list=""
results_labels=""
for run_name in "${results_order[@]}"; do
    results_list+="${OUTPUT_PATH}/${run_name}_averaged_results_${TIMESTAMP}.csv "
    results_labels+="${run_name_list[$run_name]} "
done
python3 utils/compare_results.py \
    -f ${results_list} \
    -l ${results_labels} \
    -g  synthetic_vs_real_world_and_pipelined_vs_feedback_loop_by_size \
    -o "${OUTPUT_PATH}/Results_across_all_algorithms_${TIMESTAMP}.xlsx"