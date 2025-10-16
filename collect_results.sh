#!/bin/bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Hand placed benchmarks
rm -rf ./build
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --placed-iron --run --output-dir=./build -j 1 --verbose
python3 utils/parse_results.py --variant=std --output-csv="./Results/hand_placed_results_${TIMESTAMP}.csv"
mv ./build ./Results/build_hand_placed_results_${TIMESTAMP}

rm -rf ./build
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --placed-iron --run --output-dir=./build -j 1 --verbose --aie-pkt-routing
python3 utils/parse_results.py --variant=std --output-csv="./Results/hand_placed_using_packet_flow_results_${TIMESTAMP}.csv"
mv ./build ./Results/build_hand_placed_using_packet_flow_results_${TIMESTAMP}

# AMD Sequential Placer
rm -rf ./build
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --no-placed-iron --iron-placer=sequential_placer --run --output-dir=./build -j 1 --verbose
python3 utils/parse_results.py --variant=std --output-csv="./Results/sequential_placer_results_${TIMESTAMP}.csv"
mv ./build ./Results/build_sequential_placer_results_${TIMESTAMP}

rm -rf ./build
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --no-placed-iron --iron-placer=sequential_placer --run --output-dir=./build -j 1 --verbose --aie-pkt-routing
python3 utils/parse_results.py --variant=std --output-csv="./Results/sequential_placer_using_packet_flow_results_${TIMESTAMP}.csv"
mv ./build ./Results/build_sequential_placer_using_packet_flow_results_${TIMESTAMP}

for n in 0 1 10 100 1000
do
    # SAPlacer without packing, varying n
    rm -rf ./build
    rm -rf ./tmp-parallel-pnr
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --placed-iron --no-compile --pnr --pnr-args="-n 0 -r -u sa" --output-dir=./build -j 1 --verbose
    python3 utils/parallel_pnr.py ./build -o ./tmp-parallel-pnr --pnr-args="-n $n" -j 20
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --import-pnr-results ./tmp-parallel-pnr --run -j 1 --verbose --output-dir=./build
    python3 utils/parse_results.py --variant=pnr --output-csv="./Results/saplacer_without_packing_n${n}_results_${TIMESTAMP}.csv"
    mv ./build ./Results/build_saplacer_without_packing_n${n}_${TIMESTAMP}
    mv ./tmp-parallel-pnr ./Results/tmp_parallel_pnr_saplacer_without_packing_n${n}_${TIMESTAMP}

    # SAPlacer with packing, varying n
    rm -rf ./build
    rm -rf ./tmp-parallel-pnr
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --no-placed-iron --iron-placer=sa_placer --no-compile --pnr --pnr-args="-n 0 -r -u sa" --output-dir=./build -j 1 --verbose
    python3 utils/parallel_pnr.py ./build -o ./tmp-parallel-pnr --pnr-args="-n $n" -j 20
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --import-pnr-results ./tmp-parallel-pnr --run -j 1 --verbose --output-dir=./build
    python3 utils/parse_results.py --variant=pnr --output-csv="./Results/saplacer_with_packing_n${n}_results_${TIMESTAMP}.csv"
    mv ./build ./Results/build_saplacer_with_packing_n${n}_${TIMESTAMP}
    mv ./tmp-parallel-pnr ./Results/tmp_parallel_pnr_saplacer_with_packing_n${n}_${TIMESTAMP}

    # MLIPPlacer without packing, varying n
    rm -rf ./build
    rm -rf ./tmp-parallel-pnr
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --placed-iron --no-compile --pnr --pnr-args="-n 0 -r -u milp" --output-dir=./build -j 1 --verbose
    python3 utils/parallel_pnr.py ./build -o ./tmp-parallel-pnr --pnr-args="-n $n" -j 20
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --import-pnr-results ./tmp-parallel-pnr --run -j 1 --verbose --output-dir=./build
    python3 utils/parse_results.py --variant=pnr --output-csv="./Results/milpplacer_without_packing_n${n}_results_${TIMESTAMP}.csv"
    mv ./build ./Results/build_milpplacer_without_packing_n${n}_${TIMESTAMP}
    mv ./tmp-parallel-pnr ./Results/tmp_parallel_pnr_milpplacer_without_packing_n${n}_${TIMESTAMP}

    # MLIPPlacer with packing, varying n
    rm -rf ./build
    rm -rf ./tmp-parallel-pnr
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --no-placed-iron --iron-placer=sa_placer --no-compile --pnr --pnr-args="-n 0 -r -u milp" --output-dir=./build -j 1 --verbose
    python3 utils/parallel_pnr.py ./build -o ./tmp-parallel-pnr --pnr-args="-n $n" -j 20
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --import-pnr-results ./tmp-parallel-pnr --run -j 1 --verbose --output-dir=./build
    python3 utils/parse_results.py --variant=pnr --output-csv="./Results/milpplacer_with_packing_n${n}_results_${TIMESTAMP}.csv"
    mv ./build ./Results/build_milpplacer_with_packing_n${n}_${TIMESTAMP}
    mv ./tmp-parallel-pnr ./Results/tmp_parallel_pnr_milpplacer_with_packing_n${n}_${TIMESTAMP}
done

python3 utils/compare_results.py \
    -f  Results/hand_placed_results_${TIMESTAMP}.csv \
        Results/hand_placed_using_packet_flow_results_${TIMESTAMP}.csv \
        Results/sequential_placer_results_${TIMESTAMP}.csv \
        Results/sequential_placer_using_packet_flow_results_${TIMESTAMP}.csv \
        Results/saplacer_without_packing_n0_results_${TIMESTAMP}.csv \
        Results/saplacer_without_packing_n1_results_${TIMESTAMP}.csv \
        Results/saplacer_without_packing_n10_results_${TIMESTAMP}.csv \
        Results/saplacer_without_packing_n100_results_${TIMESTAMP}.csv \
        Results/saplacer_without_packing_n1000_results_${TIMESTAMP}.csv \
        Results/saplacer_with_packing_n0_results_${TIMESTAMP}.csv \
        Results/saplacer_with_packing_n1_results_${TIMESTAMP}.csv \
        Results/saplacer_with_packing_n10_results_${TIMESTAMP}.csv \
        Results/saplacer_with_packing_n100_results_${TIMESTAMP}.csv \
        Results/saplacer_with_packing_n1000_results_${TIMESTAMP}.csv \
        Results/milpplacer_without_packing_n0_results_${TIMESTAMP}.csv \
        Results/milpplacer_without_packing_n1_results_${TIMESTAMP}.csv \
        Results/milpplacer_without_packing_n10_results_${TIMESTAMP}.csv \
        Results/milpplacer_without_packing_n100_results_${TIMESTAMP}.csv \
        Results/milpplacer_without_packing_n1000_results_${TIMESTAMP}.csv \
        Results/milpplacer_with_packing_n0_results_${TIMESTAMP}.csv \
        Results/milpplacer_with_packing_n1_results_${TIMESTAMP}.csv \
        Results/milpplacer_with_packing_n10_results_${TIMESTAMP}.csv \
        Results/milpplacer_with_packing_n100_results_${TIMESTAMP}.csv \
        Results/milpplacer_with_packing_n1000_results_${TIMESTAMP}.csv \
    -l  HP \
        HP_wPacket \
        SP \
        SP_wPacket \
        SAP_nopack_n0 \
        SAP_nopack_n1 \
        SAP_nopack_n10 \
        SAP_nopack_n100 \
        SAP_nopack_n1000 \
        SAP_pack_n0 \
        SAP_pack_n1 \
        SAP_pack_n10 \
        SAP_pack_n100 \
        SAP_pack_n1000 \
        MILPP_nopack_n0 \
        MILPP_nopack_n1 \
        MILPP_nopack_n10 \
        MILPP_nopack_n100 \
        MILPP_nopack_n1000 \
        MILPP_pack_n0 \
        MILPP_pack_n1 \
        MILPP_pack_n10 \
        MILPP_pack_n100 \
        MILPP_pack_n1000 \
    -o  overall_results_${TIMESTAMP}.xlsx