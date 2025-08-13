#!/bin/bash

trace_file_name="${1:+${1}_}"

timestamp=$(date +"%Y%m%d_%H%M%S")
trace_file_name="${trace_file_name}${timestamp}"

nsys profile -t nvtx,cuda --output=${trace_file_name} torchrun --nproc_per_node 2 train.py

pid_rank0=$(<pid_rank0.txt)
pid_rank1=$(<pid_rank1.txt)
echo "PID of rank 0: ${pid_rank0}"
echo "PID of rank 1: ${pid_rank1}"

nsys stats --report cuda_api_trace --force-export=true --output . --format json ${trace_file_name}.nsys-rep
nsys stats --report cuda_gpu_trace --force-export=true --output . --format json ${trace_file_name}.nsys-rep

if [[ ! -f ${trace_file_name}_cuda_api_trace.json ]]; then
    echo "Error: File '${trace_file_name}_cuda_api_trace.json' does not exist." >&2
    rm pid_rank0.txt
    rm pid_rank1.txt
    exit 1
fi

if [[ ! -f ${trace_file_name}_cuda_gpu_trace.json ]]; then
    echo "Error: File '${trace_file_name}_cuda_gpu_trace.json' does not exist." >&2
    rm pid_rank0.txt
    rm pid_rank1.txt
    exit 1
fi

python merge_cuda_timeline.py --rank0-pid ${pid_rank0} --rank1-pid ${pid_rank1} \
  --api-file ${trace_file_name}_cuda_api_trace.json \
  --gpu-file ${trace_file_name}_cuda_gpu_trace.json \
  --out ${trace_file_name}_merged_cuda_timeline.json

# clean up
rm pid_rank0.txt
rm pid_rank1.txt
rm ${trace_file_name}_cuda_api_trace.json
rm ${trace_file_name}_cuda_gpu_trace.json
rm ${trace_file_name}.sqlite
