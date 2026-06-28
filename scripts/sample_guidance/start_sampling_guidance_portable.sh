#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${INPUT_DIR:-/media/damoxing/datasets/astribot_tasks/myendless}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8877}"
MODEL="${MODEL:-$SCRIPT_DIR/calibration/camera_model_shared_tool.json}"
ALIGNMENT="${ALIGNMENT:-$SCRIPT_DIR/calibration/urdf_fk_alignment_right_arm.json}"
URDF="${URDF:-$SCRIPT_DIR/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body.urdf}"
RIGHT_ARM_URDF="${RIGHT_ARM_URDF:-$SCRIPT_DIR/astribot_descriptions/urdf/astribot_s1_urdf/astribot_arm_right.urdf}"
BACKGROUND="${BACKGROUND:-$SCRIPT_DIR/image.png}"
SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
SAMPLE_ATTEMPTS="${SAMPLE_ATTEMPTS:-60000}"
SAMPLE_CACHE="${SAMPLE_CACHE:-$SCRIPT_DIR/sampling_guidance_pair_cache.json}"
PRECOMPUTE_COUNT="${PRECOMPUTE_COUNT:-100}"
CENTRIFUGE_COLLECTION_DIR="${CENTRIFUGE_COLLECTION_DIR:-$SCRIPT_DIR/collection_inbox/centrifuge}"
MULTIDROP_COLLECTION_DIR="${MULTIDROP_COLLECTION_DIR:-$SCRIPT_DIR/collection_inbox/multidrop}"
COLLECTION_POSITION_TOLERANCE_M="${COLLECTION_POSITION_TOLERANCE_M:-0.03}"
COLLECTION_ROTATION_TOLERANCE_DEG="${COLLECTION_ROTATION_TOLERANCE_DEG:-20}"
COLLECTION_SCAN_INTERVAL_S="${COLLECTION_SCAN_INTERVAL_S:-0.8}"
COLLECTION_FILE_STABLE_S="${COLLECTION_FILE_STABLE_S:-1.0}"
COLLECTION_READ_TIMEOUT_S="${COLLECTION_READ_TIMEOUT_S:-10.0}"
WORKSPACE_XY="${WORKSPACE_XY:-0.18,0.65,-0.45,0.20}"
GRASP_HEIGHT="${GRASP_HEIGHT:-0.7821216622438248}"
CENTRIFUGE_PLACE_HEIGHT="${CENTRIFUGE_PLACE_HEIGHT:-0.8658671177760737}"
MULTIDROP_PLACE_HEIGHT="${MULTIDROP_PLACE_HEIGHT:-0.7886867696617663}"
OBJECT_LENGTH_CM="${OBJECT_LENGTH_CM:-13}"
OBJECT_WIDTH_CM="${OBJECT_WIDTH_CM:-8}"
BLOCK_CORRIDOR_CM="${BLOCK_CORRIDOR_CM:-7}"
MIN_TOOL_DIRECTION_DOT="${MIN_TOOL_DIRECTION_DOT:-0.0}"
RIGHT_SHOULDER_XY="${RIGHT_SHOULDER_XY:-0.0,-0.22}"

/root/miniforge/bin/python "$SCRIPT_DIR/sampling_guidance_server.py" \
  --input-dir "$INPUT_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --alignment "$ALIGNMENT" \
  --urdf "$URDF" \
  --right-arm-urdf "$RIGHT_ARM_URDF" \
  --background "$BACKGROUND" \
  --sample-count "$SAMPLE_COUNT" \
  --sample-attempts "$SAMPLE_ATTEMPTS" \
  --sample-cache "$SAMPLE_CACHE" \
  --precompute-count "$PRECOMPUTE_COUNT" \
  --centrifuge-collection-dir "$CENTRIFUGE_COLLECTION_DIR" \
  --multidrop-collection-dir "$MULTIDROP_COLLECTION_DIR" \
  --collection-position-tolerance-m "$COLLECTION_POSITION_TOLERANCE_M" \
  --collection-rotation-tolerance-deg "$COLLECTION_ROTATION_TOLERANCE_DEG" \
  --collection-scan-interval-s "$COLLECTION_SCAN_INTERVAL_S" \
  --collection-file-stable-s "$COLLECTION_FILE_STABLE_S" \
  --collection-read-timeout-s "$COLLECTION_READ_TIMEOUT_S" \
  --workspace-xy "$WORKSPACE_XY" \
  --grasp-height "$GRASP_HEIGHT" \
  --centrifuge-place-height "$CENTRIFUGE_PLACE_HEIGHT" \
  --multidrop-place-height "$MULTIDROP_PLACE_HEIGHT" \
  --object-length-cm "$OBJECT_LENGTH_CM" \
  --object-width-cm "$OBJECT_WIDTH_CM" \
  --block-corridor-cm "$BLOCK_CORRIDOR_CM" \
  --min-tool-direction-dot "$MIN_TOOL_DIRECTION_DOT" \
  --right-shoulder-xy "$RIGHT_SHOULDER_XY"
