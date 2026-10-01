#!/usr/bin/env bash
set -euo pipefail

# Batch convert ONNX models to TensorRT engine.
#
# Usage:
#   ./batch_onnx2engine.sh                          # convert all *.onnx in MODELS_DIR
#   ./batch_onnx2engine.sh model.onnx               # one file (relative to MODELS_DIR or absolute path)
#   ./batch_onnx2engine.sh a.onnx b.onnx            # multiple files
#   ./batch_onnx2engine.sh --onnx /path/to/x.onnx   # explicit path
#
# Rule:
# - Filenames containing "_1920_" are converted with 1280 input.
# - Filenames containing "_1280_" keep 1280 input.
# - Filenames containing "_640_" keep 640 input.
# - OBB models (name contains "-obb_") use dedicated TensorRT args.

MODELS_DIR="${MODELS_DIR:-/app/models}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/app/TensorRT-10.13.2.6/bin/trtexec}"
TRT_ROOT="${TRT_ROOT:-}"
TRT_LIB="${TRT_LIB:-}"
# 容器里常见: TensorRT_DIR=/app/TensorRT-10.13.2.6/targets/x86_64-linux-gnu
TRT_DIR="${TRT_DIR:-${TensorRT_DIR:-}}"
MAX_BATCH="${MAX_BATCH:-8}"
PRECISION_FLAG="${PRECISION_FLAG:---fp16}"
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"

find_trt_lib_dir() {
  local trt_root="$1"
  local candidates=()
  if [[ -n "${TRT_LIB}" ]]; then
    candidates+=("${TRT_LIB}")
  fi
  if [[ -n "${TRT_DIR}" ]]; then
    candidates+=("${TRT_DIR}/lib")
  fi
  candidates+=(
    "${trt_root}/targets/x86_64-linux-gnu/lib"
    "${trt_root}/lib"
    "/usr/lib/x86_64-linux-gnu"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${c}/libnvinfer_plugin.so.10" || -f "${c}/libnvinfer.so.10" ]]; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
  for c in "${candidates[@]}"; do
    if [[ -d "${c}" ]]; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
  return 1
}

setup_trt_library_path() {
  local trt_root="${TRT_ROOT}"
  if [[ -z "${trt_root}" ]]; then
    trt_root="$(dirname "$(dirname "$(realpath "${TRTEXEC_BIN}")")")"
  fi
  local trt_lib
  if ! trt_lib="$(find_trt_lib_dir "${trt_root}")"; then
    echo "ERROR: TensorRT lib not found under ${trt_root}" >&2
    echo "  export LD_LIBRARY_PATH=/app/TensorRT-10.13.2.6/targets/x86_64-linux-gnu/lib:\$LD_LIBRARY_PATH" >&2
    echo "  or: export TRT_LIB=/app/TensorRT-10.13.2.6/targets/x86_64-linux-gnu/lib" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${trt_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "LD_LIBRARY_PATH (+TensorRT): ${trt_lib}"
}

usage() {
  cat << 'EOF'
Usage:
  batch_onnx2engine.sh [OPTIONS] [ONNX ...]

  With no ONNX arguments: convert all *.onnx under MODELS_DIR (default /app/models).
  With ONNX arguments: convert only the listed file(s).

Arguments:
  ONNX                  Basename under MODELS_DIR, or absolute/relative path to .onnx

Options:
  --onnx PATH           Specify one ONNX path (repeatable)
  -h, --help            Show this help

Environment:
  MODELS_DIR, TRTEXEC_BIN, TRT_ROOT, TRT_DIR, TensorRT_DIR, TRT_LIB, MAX_BATCH, ...
  (auto-detect lib: targets/x86_64-linux-gnu/lib or TensorRT_DIR/lib)

Examples:
  ./batch_onnx2engine.sh fisher_visible_wide_yolo11s_1280.onnx
  ./batch_onnx2engine.sh --onnx /app/models/a.onnx --onnx /app/models/b.onnx
  MODELS_DIR=/mnt/models ./batch_onnx2engine.sh my_model_1280.onnx
EOF
}

resolve_onnx_path() {
  local arg="$1"
  if [[ -f "${arg}" ]]; then
    printf '%s\n' "$(realpath "${arg}")"
    return 0
  fi
  if [[ -f "${MODELS_DIR}/${arg}" ]]; then
    printf '%s\n' "$(realpath "${MODELS_DIR}/${arg}")"
    return 0
  fi
  if [[ "${arg}" != *.onnx ]] && [[ -f "${MODELS_DIR}/${arg}.onnx" ]]; then
    printf '%s\n' "$(realpath "${MODELS_DIR}/${arg}.onnx")"
    return 0
  fi
  echo "ERROR: ONNX not found: ${arg} (tried cwd, ${MODELS_DIR}/)" >&2
  return 1
}

declare -a onnx_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --onnx)
      shift
      [[ $# -gt 0 ]] || {
        echo "ERROR: --onnx requires a path" >&2
        exit 1
      }
      onnx_args+=("$1")
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        onnx_args+=("$1")
        shift
      done
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      onnx_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "ERROR: models directory not found: ${MODELS_DIR}" >&2
  exit 1
fi

if [[ ! -x "${TRTEXEC_BIN}" ]]; then
  if command -v trtexec > /dev/null 2>&1; then
    TRTEXEC_BIN="$(command -v trtexec)"
  else
    echo "ERROR: trtexec not found. Set TRTEXEC_BIN to a valid executable." >&2
    exit 1
  fi
fi

setup_trt_library_path

shopt -s nullglob
declare -a onnx_files=()

if [[ ${#onnx_args[@]} -gt 0 ]]; then
  for arg in "${onnx_args[@]}"; do
    resolved="$(resolve_onnx_path "${arg}")" || exit 1
    onnx_files+=("${resolved}")
  done
  echo "Target ONNX (${#onnx_files[@]}):"
  for f in "${onnx_files[@]}"; do
    echo "  - ${f}"
  done
else
  onnx_files=("${MODELS_DIR}"/*.onnx)
fi

if [[ ${#onnx_files[@]} -eq 0 ]]; then
  echo "No .onnx files to convert."
  exit 0
fi

if ! [[ "${PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
  echo "ERROR: PARALLEL_JOBS must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi

skip_count=0
task_count=0
declare -a tasks=()

for onnx_path in "${onnx_files[@]}"; do
  base_name="$(basename "${onnx_path}")"

  input_size=""
  if [[ "${base_name}" == *_1920_* ]]; then
    input_size="1280"
  elif [[ "${base_name}" == *_1280_* ]]; then
    input_size="1280"
  elif [[ "${base_name}" == *_640_* ]]; then
    input_size="640"
  else
    echo "[SKIP] ${base_name} (cannot infer input size from filename)"
    skip_count=$((skip_count + 1))
    continue
  fi

  is_obb=0
  if [[ "${base_name}" == *-obb_* ]]; then
    is_obb=1
  fi

  tasks+=("${onnx_path}|${input_size}|${is_obb}")
  task_count=$((task_count + 1))
done

if [[ ${task_count} -eq 0 ]]; then
  echo "No convertible ONNX files found. skipped=${skip_count}"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Start converting ${task_count} models with PARALLEL_JOBS=${PARALLEL_JOBS} ..."

running=0
idx=0
for task in "${tasks[@]}"; do
  IFS='|' read -r onnx_path input_size is_obb <<< "${task}"
  idx=$((idx + 1))
  status_file="${tmp_dir}/${idx}.status"

  (
    base_name="$(basename "${onnx_path}")"
    engine_path="${onnx_path%.onnx}.engine"
    shape_min="images:1x3x${input_size}x${input_size}"
    shape_opt="images:1x3x${input_size}x${input_size}"
    shape_max="images:${MAX_BATCH}x3x${input_size}x${input_size}"

    cmd=(
      "${TRTEXEC_BIN}"
      "--onnx=${onnx_path}"
      "--saveEngine=${engine_path}"
      "--minShapes=${shape_min}"
      "--optShapes=${shape_opt}"
      "--maxShapes=${shape_max}"
    )

    if [[ "${is_obb}" == "1" ]]; then
      cmd+=("--builderOptimizationLevel=0" "--precisionConstraints=none")
      echo "[RUN ] ${base_name} -> $(basename "${engine_path}") (input=${input_size}, format=obb)"
    else
      if [[ -n "${PRECISION_FLAG}" ]]; then
        cmd+=("${PRECISION_FLAG}")
      fi
      echo "[RUN ] ${base_name} -> $(basename "${engine_path}") (input=${input_size}, format=default)"
    fi

    set +e
    "${cmd[@]}"
    rc=$?
    set -e
    echo "${rc}" > "${status_file}"

    if [[ ${rc} -eq 0 ]]; then
      echo "[ OK ] ${base_name}"
    else
      echo "[FAIL] ${base_name} (exit=${rc})" >&2
    fi
  ) &

  running=$((running + 1))
  if [[ ${running} -ge ${PARALLEL_JOBS} ]]; then
    wait -n || true
    running=$((running - 1))
  fi
done

wait || true

ok_count=0
fail_count=0
for status_file in "${tmp_dir}"/*.status; do
  rc="$(< "${status_file}")"
  if [[ "${rc}" == "0" ]]; then
    ok_count=$((ok_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
done

echo "Done. success=${ok_count}, skipped=${skip_count}, failed=${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
  exit 2
fi
