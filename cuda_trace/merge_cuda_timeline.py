import argparse, json, re
from typing import Any, Dict, List, NamedTuple


RANK_RE = re.compile(r"\((\d+)\)$")


class _Identifier(NamedTuple):
    corrId: Any
    rank: Any


def extract_rank(device_str: str) -> int | None:
    if not isinstance(device_str, str):
        return None
    m = RANK_RE.search(device_str.strip())
    return int(m.group(1)) if m else None

def norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "").replace("_", "")

def keyfind(d: Dict[str, Any], key: str, default=None):
    if not d:
        return default
    for k, v in d.items():
        if norm_key(k) == norm_key(key):
            return v
    return default

def load_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data

def build_kernel_index(gpu_rows: List[Dict[str, Any]]) -> Dict[_Identifier, Dict[str, Any]]:
    kernel_index = {}
    for r in gpu_rows:
        corrId = keyfind(r, "CorrId")
        device = keyfind(r, "Device")
        if not (corrId and device):
            continue
        rank = extract_rank(device)
        if rank == None:
            raise ValueError(f"Could not extract rank from device string: {device}")
        identifier = _Identifier(corrId, rank)
        kernel_index.setdefault(identifier, []).append(r)

    for identifier, entry in kernel_index.items():
        if len(entry) > 1:
            raise ValueError(f"Multiple GPU rows found for Identifier {identifier}: {entry}")
        kernel_index[identifier] = entry[0]

    return kernel_index

def main(args):
    rank0_pid = args.rank0_pid
    rank1_pid = args.rank1_pid
    global pid_to_rank
    pid_to_rank = {rank0_pid: 0, rank1_pid: 1}

    api_entries = load_json_rows(args.api_file)
    gpu_entries = load_json_rows(args.gpu_file)
    kernel_index = build_kernel_index(gpu_entries)

    out_entries: List[Dict[str, Any]] = []
    for api_entry in api_entries:
        corrId = keyfind(api_entry, "CorrId")
        pid = keyfind(api_entry, "Pid")
        api_name = keyfind(api_entry, "Name")
        is_kernel_launch = (api_name == "cudaLaunchKernel" or api_name == "cuLaunchKernelEx")
        api_rank = pid_to_rank.get(pid, -1)
        if api_rank == -1:
            raise ValueError(f"Unknown PID {pid} in GPU rows, expected rank 0 or 1")
        identifier = _Identifier(corrId, api_rank)

        if is_kernel_launch and corrId is not None:
            kernel = kernel_index.get(identifier)
            # if keyfind(api_entry, "Name") == "cuLaunchKernelEx":
            #     print(f"api id: {identifier}, got kernel: {kernel}")
            merged = dict(api_entry)
            merged.update({"kernel": kernel if kernel else {}})
            out_entries.append(merged)
        else:
            merged = dict(api_entry)
            merged.update({"kernel": {}})
            out_entries.append(merged)

    with open(args.out, "w") as f:
        json.dump(out_entries, f, indent=2)

    print(f"Merged {len(out_entries)} rows with {len(api_entries)} entries of CUDA API and {len(gpu_entries)} entries of GPU kernel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank0-pid", required=True, type=int, help="Rank 0 process ID")
    parser.add_argument("--rank1-pid", required=True, type=int, help="Rank 1 process ID")
    parser.add_argument("--api-file", required=True, help="cuda_api_trace JSON (from nsys stats)")
    parser.add_argument("--gpu-file", required=True, help="cuda_gpu_trace JSON (from nsys stats)")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()
    main(args)
