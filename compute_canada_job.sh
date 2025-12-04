#!/bin/bash
# ---------------------------------------------------------------------
# SLURM RESOURCE CONFIGURATION
# ---------------------------------------------------------------------
#SBATCH --time=00:25:00          # Request a run time limit of 25 minutes
#SBATCH --account=def-vaughn     # Account name for the research group
#SBATCH --mem=32G                # Request 32 GB of RAM for the node
#SBATCH --cpus-per-task=16       # Request 16 CPU cores 
#SBATCH --output=slurm-%j.out    # Save slurm job info output to file

# ---------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------
# Load the Apptainer module
module load apptainer

# WORKDIR: A temporary scratch directory for high-speed I/O (if needed)
WORKDIR=$SCRATCH/apptainer_workdir_$SLURM_JOB_ID
# OUTPUT_DIR: Where results will be saved on the host system, i.e. your home directory
OUTPUT_DIR=$HOME/output_$SLURM_JOB_ID

# Create the directories if they don't exist
mkdir -p "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------
# CONTAINER EXECUTION
# ---------------------------------------------------------------------
#   -C                   : Clears host env vars for a clean environment.
#                          Crucial so host Python/Libs don't conflict with container.
#   -B "$OUTPUT_DIR":/output 
#                        : Binds the host's OUTPUT_DIR to '/output' inside container.
#                          build_benchmarks.py writes to '/output', which will map to OUTPUT_DIR.
#   --pwd /workspace     : Sets the starting directory inside the container to '/workspace'.
apptainer run \
    -C \
    -B "$OUTPUT_DIR":/output \
    --pwd /workspace \
    $HOME/projects/def-vaughn/npu-flow-images/npu-flow-app-only-2.sif \ 
    python3 utils/build_benchmarks.py edge_detection/col_1 \
        --placer="sa_placer" \
        --pnr-args="-n 20" \
        --output-dir=/output \
        -j 16 # Make sure this matches --cpus-per-task above for optimal performance or like 2 processes per core
