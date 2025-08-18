## Usage

```sh
cd /cuda_trace
./get_cuda_trace.sh [-o output_dir] [-n trace_name_prefix] [-d] [-h]
```

The the association between `cudaStreamWaitEvent` and `cudaEventRecord` event relies on `--cuda-event-trace=true` flag of `nsys profile`. 