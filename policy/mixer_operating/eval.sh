#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if (( $# < 1 )); then
  echo "Usage: policy/<name>/eval.sh GPU_ID [random|clear] [--key value ...]" >&2
  exit 2
fi
gpu_id=$1
shift
[[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be a physical integer index' >&2; exit 2; }
setting=random
if (( $# )) && [[ "$1" != --* ]]; then setting=$1; shift; fi
[[ "$setting" == random || "$setting" == clear ]] || { echo 'setting must be random or clear' >&2; exit 2; }
repo_root=$(cd "$policy_dir/../.." && pwd)
[[ -f "$repo_root/scripts/eval_policy.py" ]] || { echo 'Install this policy overlay into the pinned RoboSynChallenge checkout first.' >&2; exit 2; }
# Vulkan uses physical GPU indices, so CUDA must retain the same numbering.
unset CUDA_VISIBLE_DEVICES
nvidia_egl=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [[ -r "$nvidia_egl" ]]; then
  unset __EGL_VENDOR_LIBRARY_DIRS
  export __EGL_VENDOR_LIBRARY_FILENAMES="$nvidia_egl"
fi
for nvidia_vk in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [[ -r "$nvidia_vk" ]]; then export VK_DRIVER_FILES="$nvidia_vk" VK_ICD_FILENAMES="$nvidia_vk"; break; fi
done
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
cd "$repo_root"
exec "${PYTHON_BIN:-python}" "$policy_dir/submission_eval.py" --config "$policy_dir/deploy_policy.yml" \
  --overrides --setting "$setting" --gpu_id "$gpu_id" --pytorch_device "cuda:$gpu_id" "$@"
