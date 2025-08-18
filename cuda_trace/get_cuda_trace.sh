#!/usr/bin/env bash
set -euo pipefail

# Defaults
outdir="./cuda_timeline"
name_prefix=""
debug=false

usage() {
  echo "Usage: $0 [-o output_dir] [-n name_prefix] [-d]" >&2
  echo "  -o  Output directory (default: ./cuda_timeline)" >&2
  echo "  -n  Trace name prefix (e.g., 'run1' -> 'run1_YYYYmmdd_HHMMSS')" >&2
  echo "  -d  Enable debug mode (passed through to merge_cuda_timeline.py)" >&2
}

# Options: -o <output_dir>, -n <name_prefix>
while getopts ":o:n:dh" opt; do
  case "$opt" in
    o) outdir="$OPTARG" ;;
    n) name_prefix="$OPTARG" ;;
    d) debug=true ;;
    h) usage; exit 0 ;;
    \?) echo "Error: Unknown option -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Error: Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# No positional args allowed
if (( $# > 0 )); then
  echo "Error: Positional arguments are not supported. Use -n for the name." >&2
  usage
  exit 1
fi

# Prepare paths
mkdir -p "$outdir"
timestamp="$(date +"%Y%m%d_%H%M%S")"
trace_prefix="${name_prefix:+${name_prefix}_}"
trace_name="${trace_prefix}${timestamp}"
basepath="${outdir%/}/${trace_name}"

echo "Output directory: $outdir"
echo "Trace base name:  $trace_name"

# Profile -> ${basepath}.nsys-rep
nsys profile -t nvtx,cuda --cuda-event-trace=true --output="$basepath" torchrun --nproc_per_node 2 train.py

nsys export -t json --output="${basepath}.json" "${basepath}.nsys-rep"

# Verify JSON exists
if [[ ! -f "${basepath}.json" ]]; then
  echo "Error: File '${basepath}.json' does not exist." >&2
  rm pid_rank0.txt pid_rank1.txt
  exit 1
fi

# Read PIDs (expected in current working dir)
pid_rank0="$(<pid_rank0.txt)"
pid_rank1="$(<pid_rank1.txt)"
echo "PID of rank 0: ${pid_rank0}"
echo "PID of rank 1: ${pid_rank1}"

# Stats JSONs into $outdir (fresh export)
nsys stats --report cuda_api_trace --force-export=true --output "$basepath" --format json "${basepath}.nsys-rep"
nsys stats --report cuda_gpu_trace --force-export=true --output "$basepath" --format json "${basepath}.nsys-rep"

api_json="${basepath}_cuda_api_trace.json"
gpu_json="${basepath}_cuda_gpu_trace.json"

if [[ ! -f "$api_json" ]]; then
  echo "Error: File '$(basename "$api_json")' does not exist in '$outdir'." >&2
  rm pid_rank0.txt pid_rank1.txt
  exit 1
fi

if [[ ! -f "$gpu_json" ]]; then
  echo "Error: File '$(basename "$gpu_json")' does not exist in '$outdir'." >&2
  rm pid_rank0.txt pid_rank1.txt
  exit 1
fi

# Merge
merge_args=(
  --rank0-pid "$pid_rank0"
  --rank1-pid "$pid_rank1"
  --api-file "$api_json"
  --gpu-file "$gpu_json"
  --trace-file "${basepath}.json"
  --out "${basepath}_merged_cuda_timeline.json"
)
if $debug; then
  merge_args+=(--debug)   # pass debug flag to Python
fi
python merge_cuda_timeline.py "${merge_args[@]}"

# Clean up
rm pid_rank0.txt pid_rank1.txt
rm "${basepath}.json" "$api_json" "$gpu_json" "${basepath}.sqlite"

echo "Merged timeline: ${basepath}_merged_cuda_timeline.json"
