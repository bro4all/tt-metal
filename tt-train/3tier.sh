#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=20
#SBATCH --mem=400000
#SBATCH --partition=debug
#SBATCH --job-name=nano_gpt_job
#SBATCH --output=nano_gpt_%j.out
#SBATCH --error=nano_gpt_%j.err

# Set environmental variables
export WORKER_COUNT=1
export AGG_COUNT=1
export OPT_COUNT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Set paths - adjust these as needed
export TT_METAL_HOME="/data/asrinivasan/tt-metal"
export TT_MESH_ID=0
export TT_HOST_RANK=0
export CONFIG="$TT_METAL_HOME/tt-train/configs/training_shakespear_nanogpt_3tier.yaml"
export BIN_DIR="$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt"
# Set run flag (empty by default, can be set to specific flags if needed)
export RUN_FLAG=""

# # Run the MPI job
mpirun -np "${WORKER_COUNT}" "${BIN_DIR}/nano_gpt" -c "${CONFIG}" \
       : -np "${AGG_COUNT}" "${BIN_DIR}/nano_gpt_aggregator" -c "${CONFIG}" \
       : -np "${OPT_COUNT}" "${BIN_DIR}/nano_gpt_optimizer" -c "${CONFIG}"
