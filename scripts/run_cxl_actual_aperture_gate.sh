#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
set -euo pipefail

usage() {
    printf '%s\n' \
        "usage: $0 --qemu FILE --qemu-data DIR --qemu-lib-dir DIR" \
        "       --server FILE --server-lib-dir DIR --virtiofsd FILE" \
        "       --cuda-library FILE --cuda-device INDEX --fixture FILE" \
        "       --file-offset BYTES --shmem-offset BYTES --length BYTES" \
        "       --model-aperture-offset BYTES" \
        "       --model-aperture-size BYTES --port PORT" \
        "       --work-dir ABSENT_DIR --output ABSENT_FILE"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf '%s\n' \
        'agent_hint=self=scripts/run_cxl_actual_aperture_gate.sh' \
        'agent_hint=boundary=production QEMU mapping/CUDA gate' \
        'agent_hint=inputs=explicit component, model, and GPU identities'
    exit 0
fi

qemu= qemu_data= qemu_lib_dir= server= server_lib_dir= virtiofsd=
cuda_library= cuda_device= fixture= file_offset= shmem_offset= length=
aperture_offset= aperture_size= port= work_dir= output=
while (( $# )); do
    (( $# >= 2 )) || { usage >&2; exit 2; }
    case "$1" in
        --qemu) qemu=$2 ;;
        --qemu-data) qemu_data=$2 ;;
        --qemu-lib-dir) qemu_lib_dir=$2 ;;
        --server) server=$2 ;;
        --server-lib-dir) server_lib_dir=$2 ;;
        --virtiofsd) virtiofsd=$2 ;;
        --cuda-library) cuda_library=$2 ;;
        --cuda-device) cuda_device=$2 ;;
        --fixture) fixture=$2 ;;
        --file-offset) file_offset=$2 ;;
        --shmem-offset) shmem_offset=$2 ;;
        --length) length=$2 ;;
        --model-aperture-offset) aperture_offset=$2 ;;
        --model-aperture-size) aperture_size=$2 ;;
        --port) port=$2 ;;
        --work-dir) work_dir=$2 ;;
        --output) output=$2 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift 2
done

for value in qemu qemu_data qemu_lib_dir server server_lib_dir virtiofsd \
             cuda_library cuda_device fixture file_offset shmem_offset length \
             aperture_offset aperture_size port work_dir output; do
    if [[ -z ${!value} ]]; then
        printf 'missing --%s\n' "${value//_/-}" >&2
        exit 2
    fi
done
for path in "$qemu" "$qemu_data" "$qemu_lib_dir" "$server" \
            "$server_lib_dir" "$virtiofsd" "$cuda_library" "$fixture" \
            "$work_dir" "$output"; do
    if [[ $path != /* ]]; then
        printf 'paths must be absolute: %s\n' "$path" >&2
        exit 2
    fi
    if [[ $path == *,* ]]; then
        printf 'QEMU property paths cannot contain commas: %s\n' "$path" >&2
        exit 2
    fi
done
for executable in "$qemu" "$server" "$virtiofsd"; do
    [[ -f $executable && -x $executable && ! -L $executable ]] || {
        printf 'executable must be a regular non-symlink file: %s\n' \
            "$executable" >&2
        exit 2
    }
done
for directory in "$qemu_data" "$qemu_lib_dir" "$server_lib_dir"; do
    [[ -d $directory && ! -L $directory ]] || {
        printf 'directory must be an existing non-symlink: %s\n' \
            "$directory" >&2
        exit 2
    }
done
[[ -f $cuda_library && ! -L $cuda_library ]] || {
    printf 'CUDA library must be an explicit regular non-symlink file\n' >&2
    exit 2
}
[[ -f $fixture && ! -L $fixture ]] || {
    printf 'fixture must be an exact regular non-symlink file\n' >&2
    exit 2
}
[[ ! -e $work_dir && ! -L $work_dir ]] || {
    printf 'work directory must not exist: %s\n' "$work_dir" >&2
    exit 2
}
[[ ! -e $output && ! -L $output ]] || {
    printf 'output must not exist: %s\n' "$output" >&2
    exit 2
}
[[ -d $(dirname "$work_dir") && -d $(dirname "$output") ]] || {
    printf 'work and output parents must exist\n' >&2
    exit 2
}
for number in "$cuda_device" "$file_offset" "$shmem_offset" "$length" \
              "$aperture_offset" "$aperture_size" "$port"; do
    if [[ ! $number =~ ^[0-9]+$ ]]; then
        printf 'numeric argument is invalid: %s\n' "$number" >&2
        exit 2
    fi
done
(( cuda_device >= 0 && port > 0 && port <= 65535 &&
   length > 0 && aperture_size > 0 )) || {
    printf 'device, port, length, or aperture size is out of range\n' >&2
    exit 2
}

readonly page_size=$(getconf PAGESIZE)
(( file_offset % page_size == 0 && shmem_offset % page_size == 0 &&
   length % page_size == 0 && aperture_offset % page_size == 0 &&
   aperture_size % page_size == 0 && length >= 10 * page_size &&
   length <= aperture_size )) || {
    printf 'file, shmem, length and aperture geometry is invalid\n' >&2
    exit 2
}
readonly fixture_size=$(stat -Lc '%s' "$fixture")
(( file_offset <= fixture_size && length <= fixture_size - file_offset )) || {
    printf 'fixture range is out of bounds\n' >&2
    exit 2
}

mkdir "$work_dir"
readonly server_log=$work_dir/cxlmemsim.log
server_pid=
virtiofsd_pid=
cleanup() {
    local pid
    for pid in "$virtiofsd_pid" "$server_pid"; do
        if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

LD_LIBRARY_PATH=$server_lib_dir "$server" \
    --comm-mode=tcp --port="$port" --capacity=64 --default_latency=100 \
    >"$server_log" 2>&1 &
server_pid=$!

wait_dependencies() {
    local socket=$1

    python3 - "$server_pid" "$virtiofsd_pid" "$socket" "$port" <<'PY'
import os
import socket
import sys
import time

server_pid, virtiofsd_pid = map(int, sys.argv[1:3])
socket_path = sys.argv[3]
port = int(sys.argv[4])
deadline = time.monotonic() + 10
last_error = "readiness timeout"
while time.monotonic() < deadline:
    for role, pid in (("cxlmemsim", server_pid), ("virtiofsd", virtiofsd_pid)):
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise SystemExit(f"{role} exited before readiness") from error
    if os.path.exists(socket_path):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError as error:
            last_error = str(error)
    time.sleep(0.05)
else:
    raise SystemExit(f"gate dependencies did not become ready: {last_error}")
PY
}

stop_virtiofsd() {
    local attempt

    for (( attempt = 0; attempt < 20; attempt++ )); do
        kill -0 "$virtiofsd_pid" 2>/dev/null || break
        sleep 0.05
    done
    if kill -0 "$virtiofsd_pid" 2>/dev/null; then
        kill "$virtiofsd_pid" 2>/dev/null || true
    fi
    virtiofsd_exit=0
    wait "$virtiofsd_pid" 2>/dev/null || virtiofsd_exit=$?
    virtiofsd_pid=
}

run_qemu_gate() {
    local mode=$1
    local result_path=$2
    local qemu_log=$3
    local virtiofsd_log=$4
    local socket=$5
    local device qemu_exit

    "$virtiofsd" \
        --shared-dir "$(dirname "$fixture")" \
        --socket-path "$socket" --sandbox none --seccomp none --readonly \
        --cache always --thread-pool-size 0 --log-level info \
        --dax-window-size 67108864 >"$virtiofsd_log" 2>&1 &
    virtiofsd_pid=$!
    wait_dependencies "$socket"

    device="cxl-type2,bus=type2_rp,id=cxl_type2_alias_gate"
    device+=",sn=205,gpu-mode=2,hetgpu-lib=$cuda_library"
    device+=",hetgpu-backend=3,hetgpu-device=$cuda_device"
    device+=",mem-size=64M,cxlmemsim-addr=127.0.0.1,cxlmemsim-port=$port"
    device+=",direct-source-fs=model_share_virtiofs_device"
    device+=",model-aperture-offset=$aperture_offset"
    device+=",model-aperture-size=$aperture_size"
    device+=",model-alias-gate-output=$result_path"
    device+=",model-alias-gate-fixture=$fixture"
    device+=",model-alias-gate-mode=$mode"
    device+=",model-alias-gate-file-offset=$file_offset"
    device+=",model-alias-gate-shmem-offset=$shmem_offset"
    device+=",model-alias-gate-length=$length"

    set +e
    LD_LIBRARY_PATH=$qemu_lib_dir HETGPU_STRICT=1 "$qemu" \
        -accel tcg,thread=multi -cpu max -nodefaults \
        -display none -serial none \
        -L "$qemu_data" -S -m 256M \
        -object memory-backend-memfd,id=guest-mem,size=256M,share=on \
        -numa node,memdev=guest-mem \
        -M q35,cxl=on,cxl-fmw.0.targets.0=cxl.0,cxl-fmw.0.size=256M \
        -device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.0 \
        -device cxl-rp,port=0,bus=cxl.0,id=type2_rp,chassis=0,slot=2 \
        -chardev "socket,id=model_share_virtiofs,path=$socket" \
        -device "vhost-user-fs-pci,id=model_share_virtiofs_device,"\
"chardev=model_share_virtiofs,tag=model,bus=pcie.0" \
        -device "$device" >"$qemu_log" 2>&1
    qemu_exit=$?
    set -e
    stop_virtiofsd
    last_qemu_exit=$qemu_exit
    last_virtiofsd_exit=$virtiofsd_exit
}

readonly read_result=$work_dir/qemu-read-gate.json
readonly read_qemu_log=$work_dir/qemu-read.log
readonly read_virtiofsd_log=$work_dir/virtiofsd-read.log
run_qemu_gate read "$read_result" "$read_qemu_log" \
    "$read_virtiofsd_log" "$work_dir/virtiofsd-read.sock"
read_qemu_exit=$last_qemu_exit
read_virtiofsd_exit=$last_virtiofsd_exit

readonly write_result=$work_dir/qemu-write-negative-gate.json
readonly write_qemu_log=$work_dir/qemu-write-negative.log
readonly write_virtiofsd_log=$work_dir/virtiofsd-write-negative.log
run_qemu_gate write-negative "$write_result" "$write_qemu_log" \
    "$write_virtiofsd_log" "$work_dir/virtiofsd-write-negative.sock"
write_qemu_exit=$last_qemu_exit
write_virtiofsd_exit=$last_virtiofsd_exit

if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
fi
wait "$server_pid" 2>/dev/null || server_exit=$?
server_pid=
server_exit=${server_exit:-0}

python3 - "$output" "$read_result" "$write_result" \
    "$read_qemu_exit" "$write_qemu_exit" "$server_exit" \
    "$read_virtiofsd_exit" "$write_virtiofsd_exit" \
    "$qemu" "$server" "$virtiofsd" "$cuda_library" "$cuda_device" \
    "$fixture" "$file_offset" "$length" "$port" "$read_qemu_log" \
    "$write_qemu_log" "$server_log" "$read_virtiofsd_log" \
    "$write_virtiofsd_log" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

(
    output, read_result, write_result, read_qemu_exit, write_qemu_exit,
    server_exit, read_virtiofsd_exit, write_virtiofsd_exit, qemu, server,
    virtiofsd, cuda_library, cuda_device, fixture, file_offset, length, port,
    read_qemu_log, write_qemu_log, server_log, read_virtiofsd_log,
    write_virtiofsd_log,
) = sys.argv[1:]

def digest(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def log_fact(path: str) -> dict[str, object]:
    candidate = Path(path)
    return {
        "path": path,
        "size": candidate.stat().st_size,
        "sha256": digest(path),
    }

try:
    gpu_lines = subprocess.run(
        [
            "nvidia-smi",
            f"--id={cuda_device}",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if len(gpu_lines) != 1:
        raise ValueError("nvidia-smi did not return exactly one GPU")
    gpu_uuid, gpu_name, gpu_memory_mib, driver_version = (
        field.strip() for field in gpu_lines[0].split(",", 3)
    )
    gpu = {
        "uuid": gpu_uuid,
        "name": gpu_name,
        "memory_mib": int(gpu_memory_mib),
        "driver_version": driver_version,
    }
except (OSError, subprocess.CalledProcessError, ValueError) as error:
    gpu = {"status": "unavailable", "reason": str(error)}

mechanisms = {}
mechanism_errors = {}
for mode, result_path in (
    ("read", read_result),
    ("write-negative", write_result),
):
    try:
        mechanism = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if (
            mechanism.get("kind") != "cxl-type2-model-alias-gate"
            or mechanism.get("mode") != mode
        ):
            raise ValueError(f"QEMU {mode} gate identity is invalid")
        mechanisms[mode] = mechanism
    except (OSError, json.JSONDecodeError, ValueError) as error:
        mechanism_errors[mode] = str(error)

passed = (
    int(read_qemu_exit) == 0
    and int(write_qemu_exit) == 0
    and int(server_exit) in (0, -15, 143)
    and int(read_virtiofsd_exit) in (0, -15, 143)
    and int(write_virtiofsd_exit) in (0, -15, 143)
    and set(mechanisms) == {"read", "write-negative"}
    and all(
        mechanism.get("status") == "pass"
        for mechanism in mechanisms.values()
    )
    and "status" not in gpu
)
first_mechanism_failure: object = next(
    (
        mechanism.get("first_failure")
        for mechanism in mechanisms.values()
        if mechanism.get("status") != "pass"
    ),
    None,
)

def failure() -> object:
    if passed:
        return None
    if first_mechanism_failure is not None:
        return first_mechanism_failure
    process_failures = (
        ("qemu-read-process", int(read_qemu_exit), {0}),
        ("qemu-write-negative-process", int(write_qemu_exit), {0}),
        ("virtiofsd-read-process", int(read_virtiofsd_exit), {0, -15, 143}),
        (
            "virtiofsd-write-negative-process",
            int(write_virtiofsd_exit),
            {0, -15, 143},
        ),
        ("cxlmemsim-process", int(server_exit), {0, -15, 143}),
    )
    for stage, result, accepted in process_failures:
        if result not in accepted:
            return {"stage": stage, "result": result}
    if mechanism_errors:
        return {"stage": "qemu-result", "reason": mechanism_errors}
    return {"stage": "gpu-identity", "reason": gpu.get("reason")}

document = {
    "kind": "cxl-type2-actual-aperture-gate",
    "status": "pass" if passed else "fail",
    "first_failure": failure(),
    "inputs": {
        "qemu": {"path": qemu, "sha256": digest(qemu)},
        "cxlmemsim": {
            "path": server,
            "sha256": digest(server),
            "port": int(port),
        },
        "virtiofsd": {"path": virtiofsd, "sha256": digest(virtiofsd)},
        "cuda_library": {"path": cuda_library, "sha256": digest(cuda_library)},
        "cuda_device": int(cuda_device),
        "gpu": gpu,
        "fixture": {
            "path": fixture,
            "sha256": digest(fixture),
            "size": os.stat(fixture).st_size,
            "file_offset": int(file_offset),
            "length": int(length),
        },
    },
    "processes": {
        "qemu_read": {"exit_code": int(read_qemu_exit)},
        "qemu_write_negative": {"exit_code": int(write_qemu_exit)},
        "cxlmemsim": {"exit_code": int(server_exit)},
        "virtiofsd_read": {"exit_code": int(read_virtiofsd_exit)},
        "virtiofsd_write_negative": {"exit_code": int(write_virtiofsd_exit)},
    },
    "logs": {
        "qemu_read": log_fact(read_qemu_log),
        "qemu_write_negative": log_fact(write_qemu_log),
        "cxlmemsim": log_fact(server_log),
        "virtiofsd_read": log_fact(read_virtiofsd_log),
        "virtiofsd_write_negative": log_fact(write_virtiofsd_log),
    },
    "mechanisms": mechanisms,
    "mechanism_errors": mechanism_errors,
}
destination = Path(output)
temporary = destination.with_name(destination.name + ".tmp")
temporary.write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, destination)
raise SystemExit(0 if passed else 1)
PY
