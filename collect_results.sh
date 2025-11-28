#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PNR_COMMIT_HASH=$(git rev-parse --short HEAD:./npu-pnr)
OUTPUT_PATH="./Results"
PNR_OUTPUT_PATH="${OUTPUT_PATH}/PnR_commit_${PNR_COMMIT_HASH}"

# Collect hand placed results
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --placer="hand_placed" --run --output-dir="${OUTPUT_PATH}/hand_placed_${TIMESTAMP}" --verbose -j 20
python utils/parse_results.py --tasklists benchmarks/tasklist.yml --input-dir "${OUTPUT_PATH}/hand_placed_${TIMESTAMP}" --output-csv "${OUTPUT_PATH}/hand_placed_${TIMESTAMP}/hand_placed_results_${TIMESTAMP}.csv"

# Collect sequential placer results
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --placer="sequential_placer" --run --output-dir="${OUTPUT_PATH}/sequential_placer_${TIMESTAMP}" --verbose -j 20
python utils/parse_results.py --tasklists benchmarks/tasklist.yml --input-dir "${OUTPUT_PATH}/sequential_placer_${TIMESTAMP}" --output-csv "${OUTPUT_PATH}/sequential_placer_${TIMESTAMP}/sequential_placer_results_${TIMESTAMP}.csv"

# Collect SAPlacer results
# Pure Greedy runs
n=1000
m=4
g=$n
RUN_NAME="SAPlacer_pure_greedy"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer="sa_placer" \
    --pnr-args="-u sa -s 0 -r -n $n -T 0 -c 0 -g $g -m $m --log-interval 1 --write-interval 100" \
    --output-dir="${PNR_OUTPUT_DIR}" \
    --verbose \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Run checkpoint iteration collections results
for iter in 100 200 300 400 500 600 700 800 900 1000
do
    output_dir="${PNR_OUTPUT_PATH}/${RUN_NAME}_n${iter}_${TIMESTAMP}"
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
        --placer="sa_placer" \
        --import-pnr-results ${PNR_OUTPUT_DIR} \
        --import-pnr-results-suffix=".json_iter_${iter}" \
        --run \
        --output-dir="${output_dir}" \
        --verbose \
        -j 20
    python utils/parse_results.py \
        --tasklists benchmarks/tasklist.yml \
        --input-dir "${output_dir}" \
        --output-csv "${output_dir}/${RUN_NAME}_n${iter}_results_${TIMESTAMP}.csv"
done

# Dynamic Temperature Scheduling runs
n=1000
m=4
g=200
RUN_NAME="SAPlacer_dynamic_temperature_scheduling"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer="sa_placer" \
    --pnr-args="-u sa -s 0 -r -n $n --dynamic-temperature-scheduling -g $g -m $m --log-interval 1 --write-interval 100" \
    --output-dir="${PNR_OUTPUT_DIR}" \
    --verbose \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Run checkpoint iteration collections results
for iter in 100 200 300 400 500 600 700 800 900 1000
do
    output_dir="${PNR_OUTPUT_PATH}/${RUN_NAME}_n${iter}_${TIMESTAMP}"
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
        --placer="sa_placer" \
        --import-pnr-results ${PNR_OUTPUT_DIR} \
        --import-pnr-results-suffix=".json_iter_${iter}" \
        --run \
        --output-dir="${output_dir}" \
        --verbose \
        -j 20
    python utils/parse_results.py \
        --tasklists benchmarks/tasklist.yml \
        --input-dir "${output_dir}" \
        --output-csv "${output_dir}/${RUN_NAME}_n${iter}_results_${TIMESTAMP}.csv"
done

# Collect LSMOPlacer results
n=1000
RUN_NAME="LSMOPlacer"
PNR_OUTPUT_DIR="${PNR_OUTPUT_PATH}/${RUN_NAME}_${TIMESTAMP}"
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
    --placer="sa_placer" \
    --pnr-args="-u lsmo -s 0 -r -n $n -g 200 -m $m --log-interval 1 --write-interval 100" \
    --output-dir="${PNR_OUTPUT_DIR}" \
    --verbose \
    -j 20
python utils/parse_results.py \
    --tasklists benchmarks/tasklist.yml \
    --input-dir "${PNR_OUTPUT_DIR}" \
    --output-csv "${PNR_OUTPUT_DIR}/${RUN_NAME}_results_${TIMESTAMP}.csv"
# Run checkpoint iteration collections results
for iter in 100 200 300 400 500 600 700 800 900 1000
do
    output_dir="${PNR_OUTPUT_PATH}/${RUN_NAME}_n${iter}_${TIMESTAMP}"
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
        --placer="sa_placer" \
        --import-pnr-results ${PNR_OUTPUT_DIR} \
        --import-pnr-results-suffix=".json_iter_${iter}" \
        --run \
        --output-dir="${output_dir}" \
        --verbose \
        -j 20
    python utils/parse_results.py \
        --tasklists benchmarks/tasklist.yml \
        --input-dir "${output_dir}" \
        --output-csv "${output_dir}/${RUN_NAME}_n${iter}_results_${TIMESTAMP}.csv"
done
