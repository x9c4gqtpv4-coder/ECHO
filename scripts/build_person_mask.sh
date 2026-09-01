#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
source_file="$project_dir/tools/person_mask/main.swift"
output_dir="$project_dir/tools/person_mask/bin"
output_file="$output_dir/batch-color-person-mask"

mkdir -p "$output_dir"
xcrun swiftc -O -parse-as-library "$source_file" -o "$output_file"
if [[ -d "$project_dir/.venv/bin" ]]; then
  ln -sf "$output_file" "$project_dir/.venv/bin/batch-color-person-mask"
fi
echo "$output_file"
