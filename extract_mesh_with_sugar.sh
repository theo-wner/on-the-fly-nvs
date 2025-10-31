#!/bin/bash
# Usage:
#   bash extract_mesh_with_sugar.sh --dataset ~/on-the-fly-nvs/results/000033

# Exit on error
set -e

# Initialize variables
ORIGIN_DATASET=""
RESULT_DATASET=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --origin_dataset)
            ORIGIN_DATASET="$2"
            shift 2
            ;;
        --result_dataset)
            RESULT_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --origin_dataset <path> --result_dataset <path>"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$ORIGIN_DATASET" || -z "$RESULT_DATASET" ]]; then
    echo "Error: both --origin_dataset and --result_dataset are required."
    echo "Usage: $0 --origin_dataset <path> --result_dataset <path>"
    exit 1
fi

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sugar
cd ~/SuGaR

# Run the crop script
python crop.py --input_path "${RESULT_DATASET}/point_clouds" --abs 3

# Create required data structure
mkdir -p "${RESULT_DATASET}/gs_checkpoint/point_cloud/iteration_7000"

# Copy files
cp "${RESULT_DATASET}/point_clouds/cropped.ply" "${RESULT_DATASET}/gs_checkpoint/point_cloud/iteration_7000/point_cloud.ply"
cp -r "${ORIGIN_DATASET}/images" "${RESULT_DATASET}/colmap/"

# Run the colmap2json script
python colmap2json.py \
 --images "${RESULT_DATASET}/colmap/images.bin" \
 --cameras "${RESULT_DATASET}/colmap/cameras.bin" \
 --images_folder "${ORIGIN_DATASET}/images" \
 --output "${RESULT_DATASET}/gs_checkpoint/cameras.json"

# Extract mesh
python extract_mesh.py \
 -s "${RESULT_DATASET}/colmap" \
 -c "${RESULT_DATASET}/gs_checkpoint/" \
 -i 7000 \
 -o "${RESULT_DATASET}/mesh" \
 --use_vanilla_3dgs True