import argparse, json, re
from typing import Any, Dict, List, NamedTuple, Optional
import logging


TYPE_CUDA_EVENT_RECORD = 127
TYPE_CUDA_EVENT_SYNC = 106
RANK_RE = re.compile(r"\((\d+)\)$")

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class _KernelIdentifier(NamedTuple):
    corr_id: Any
    rank: Any


class _EventRecordIdentifier(NamedTuple):
    device_id: Any
    event_id: Any
    event_sync_id: Any


def extract_rank(device_str: str) -> int | None:
    if not isinstance(device_str, str):
        return None
    m = RANK_RE.search(device_str.strip())
    return int(m.group(1)) if m else None

def norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "").replace("_", "")

def keyfind(d: Dict[str, Any], key: str, default=None) -> Any:
    if not d:
        return default
    for k, v in d.items():
        if norm_key(k) == norm_key(key):
            return v
    return default

def find_closest_event(target_start_time, entries) -> Optional[Dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    
    closest_event = None
    closest_diff = None

    for entry in entries:
        entry_type = keyfind(entry, "Type")
        assert entry_type == TYPE_CUDA_EVENT_SYNC or entry_type == TYPE_CUDA_EVENT_RECORD
        cuda_event = keyfind(entry, "CudaEvent")
        event_start_time = int(keyfind(cuda_event, "startNs"))
        if event_start_time is None:
            continue
        diff = abs(event_start_time - target_start_time)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_event = entry

    return closest_event

def load_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data

def load_ndjson_rows(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data

def build_kernel_index(gpu_rows: List[Dict[str, Any]]) -> Dict[_KernelIdentifier, Dict[str, Any]]:
    kernel_index = {}
    for r in gpu_rows:
        corr_id = keyfind(r, "CorrId")
        device = keyfind(r, "Device")
        if not (corr_id and device):
            continue
        rank = extract_rank(device)
        if rank == None:
            raise ValueError(f"Could not extract rank from device string: {device}")
        identifier = _KernelIdentifier(corr_id, rank)
        kernel_index.setdefault(identifier, []).append(r)

    for identifier, entry in kernel_index.items():
        if len(entry) > 1:
            raise ValueError(f"Multiple GPU rows found for Identifier {identifier}: {entry}")
        kernel_index[identifier] = entry[0]
    return kernel_index

def build_cuda_event_index(trace_rows: List[Dict[str, Any]]) -> Dict[int, Dict[int, Any]]:
    type_int_set = set()
    cuda_event_index = {TYPE_CUDA_EVENT_RECORD: {}, TYPE_CUDA_EVENT_SYNC: {}}
    for r in trace_rows:
        type_int = keyfind(r, "Type")
        type_int_set.add(type_int)
        if not (type_int == TYPE_CUDA_EVENT_RECORD or type_int == TYPE_CUDA_EVENT_SYNC):
            continue
        cuda_event = keyfind(r, "CudaEvent")
        corr_id = cuda_event.get("correlationId")
        cuda_event_index[type_int].setdefault(corr_id, []).append(r)

    logger.debug(f"Found CUDA event types: {type_int_set}")

    return cuda_event_index

def main(args):
    rank0_pid = args.rank0_pid
    rank1_pid = args.rank1_pid
    pid_to_rank = {rank0_pid: 0, rank1_pid: 1}

    api_entries = load_json_rows(args.api_file)
    gpu_entries = load_json_rows(args.gpu_file)
    trace_entries = load_ndjson_rows(args.trace_file)

    kernel_index = build_kernel_index(gpu_entries)
    cuda_event_index = build_cuda_event_index(trace_entries)

    out_entries: Dict[int, Dict[str, Any]] = {}
    event_record_id_to_trace_index: Dict[_EventRecordIdentifier, int] = {}
    trace_index_to_event_sync_id: Dict[int, _EventRecordIdentifier] = {}
    num_wait_event = 0
    for i, api_entry in enumerate(api_entries):
        corr_id = keyfind(api_entry, "CorrId")
        pid = keyfind(api_entry, "Pid")
        api_name = keyfind(api_entry, "Name")

        is_kernel_launch = (api_name == "cudaLaunchKernel" or api_name == "cuLaunchKernelEx")
        is_record_event = (api_name == "cudaEventRecord")
        is_sync_event = (api_name == "cudaStreamWaitEvent" or api_name == "cudaStreamSynchronize")

        if is_kernel_launch and corr_id is not None:
            api_rank = pid_to_rank.get(pid, -1)
            if api_rank == -1:
                raise ValueError(f"Unknown PID {pid} in GPU rows, expected rank 0 or 1")
            kernel_identifier = _KernelIdentifier(corr_id, api_rank)

            kernel = kernel_index.get(kernel_identifier)
            assert kernel is not None, f"Could not find kernel for API entry {api_entry}"
            merged = dict(api_entry)
            merged.update({"GPU Event": kernel})
            out_entries[i] = merged
        elif is_sync_event and corr_id is not None:
            api_start_time = keyfind(api_entry, "Start (ns)")
            sync_event = find_closest_event(api_start_time, cuda_event_index[TYPE_CUDA_EVENT_SYNC].get(corr_id))
            assert sync_event is not None, f"Could not find sync event for API entry {api_entry}"
            num_wait_event += 1
            merged = dict(api_entry)
            merged.update({"GPU Event": sync_event})
            out_entries[i] = merged

            cuda_event = keyfind(sync_event, "CudaEvent")
            event_sync_info = cuda_event.get("sync")
            event_record_identifier = _EventRecordIdentifier(
                pid_to_rank.get(pid, -1),
                event_sync_info.get("eventId"),
                event_sync_info.get("eventSyncId"),
            )
            trace_index_to_event_sync_id[i] = event_record_identifier
        elif is_record_event and corr_id is not None:
            api_start_time = keyfind(api_entry, "Start (ns)")
            record_event = find_closest_event(api_start_time, cuda_event_index[TYPE_CUDA_EVENT_RECORD].get(corr_id))
            assert record_event is not None, f"Could not find record event for API entry {api_entry}"
            merged = dict(api_entry)
            merged.update({"GPU Event": record_event})
            out_entries[i] = merged

            cuda_event = keyfind(record_event, "CudaEvent")
            event_record_info = cuda_event.get("cudaEventRecord")
            event_record_identifier = _EventRecordIdentifier(
                pid_to_rank.get(pid, -1),
                event_record_info.get("eventId"),
                event_record_info.get("eventSyncId"),
            )
            event_record_id_to_trace_index[event_record_identifier] = i
        else:
            merged = dict(api_entry)
            merged.update({"GPU Event": {}})
            out_entries[i] = merged

    logger.debug(f"num wait event: {num_wait_event}")

    num_link_event = 0
    num_link_fail = 0
    for i in trace_index_to_event_sync_id:
        event_record_identifier = trace_index_to_event_sync_id[i]
        event_record_trace_index = event_record_id_to_trace_index.get(event_record_identifier, -1)
        if event_record_trace_index != -1:
            corr_id = keyfind(out_entries[event_record_trace_index], "CorrId")
            out_entries[i].update({"Sync Event CorrID": corr_id})
            num_link_event += 1
        else:
            logger.debug(f"corrId {keyfind(out_entries[i], 'CorrId')}"
                f" pid {keyfind(out_entries[i], 'Pid')}"
                f" thread id {keyfind(out_entries[i], 'Tid')}"
                f" thread name {keyfind(out_entries[i], 'Thread Name')}"
                f" event id {keyfind(keyfind(keyfind(keyfind(out_entries[i], 'GPU Event'), 'CudaEvent'), 'sync'), 'eventId')}"
                f" event id {keyfind(keyfind(keyfind(keyfind(out_entries[i], 'GPU Event'), 'CudaEvent'), 'sync'), 'eventSyncId')}"
                " doesn't link to event record")
            num_link_fail += 1

    logger.debug(f"num link event: {num_link_event}, num link fail: {num_link_fail}")

    with open(args.out, "w") as f:
        json.dump(list(out_entries.values()), f, indent=2)

    logger.info(f"Merged {len(out_entries)} rows with {len(api_entries)} entries of CUDA API and {len(gpu_entries)} entries of GPU kernel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank0-pid", required=True, type=int, help="Rank 0 process ID")
    parser.add_argument("--rank1-pid", required=True, type=int, help="Rank 1 process ID")
    parser.add_argument("--api-file", required=True, help="cuda_api_trace JSON (from nsys stats)")
    parser.add_argument("--gpu-file", required=True, help="cuda_gpu_trace JSON (from nsys stats)")
    parser.add_argument("--trace-file", required=True, help="raw trace JSON (from nsys export)")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    main(args)
