#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
output="$project_dir/models/selfie_multiclass_256x256.tflite"
expected="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

mkdir -p "$project_dir/models"
temporary="$output.download"
trap 'rm -f "$temporary"' EXIT
curl --fail --location --max-time 180 --output "$temporary" "$url"
actual="$(shasum -a 256 "$temporary" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "Model checksum mismatch: $actual" >&2
  exit 1
fi
mv "$temporary" "$output"
trap - EXIT
echo "$output"
