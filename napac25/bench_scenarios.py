#!/usr/bin/env python3
#
# Requirements in parent env:
# - conda/mamba/miniconda
# - jinja2
# - git

import os
import re
import socket
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader


# general configs
#
build_nproc = 12
conda = "mamba"
# we vary the number of particles to push in the beam,
# to see if a code can make efficient use of L1/L2/L3 caches
nparts = [1_000, 10_000, 100_000]  # TODO: add , 1_000_000, 10_000_000]:

code_configs = {
    "impactx": {
        "code": "impactx",
        "version": "development",  # 25.08
        "gh_owner": "BLAST-ImpactX",
        "CXXFLAGS": "-march=native -ffast-math",
        "ImpactX_COMPUTE": "OMP",
        "ImpactX_SIMD": "OFF",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "impactx-simd": {
        "code": "impactx",
        "version": "development",  # 25.08
        "gh_owner": "BLAST-ImpactX",
        "CXXFLAGS": "-march=native -ffast-math",
        "ImpactX_COMPUTE": "OMP",
        "ImpactX_SIMD": "ON",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "impactx-gpu": {
        "code": "impactx",
        "version": "development",  # 25.08
        "gh_owner": "BLAST-ImpactX",
        "CXXFLAGS": "",
        "ImpactX_COMPUTE": "CUDA",
        "ImpactX_SIMD": "OFF",
        "env_name": "benchmark-gpu",
        "env_file": "benchmark-gpu-conda.yaml",
    },
    "cheetah": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "none",  # https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "none",
        "device": "cpu",
        "dtype": "torch.float32",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "cheetah-compiled-default": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "default",  # TODO: try also "max-autotune" on CPUs https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "inductor",  # TODO: try also "ipex" on Intel CPUs
        "device": "cpu",
        "dtype": "torch.float32",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "cheetah-gpu": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "none",  # https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "none",
        "device": "gpu",
        "dtype": "float32",
        "env_name": "benchmark-gpu",
        "env_file": "benchmark-gpu-conda.yaml",
    },
    "cheetah-gpu-compiled-default": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "default",  # TODO: try also "max-autotune" on CPUs https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "inductor",  # TODO: try also "ipex", "onnxrt" on (Intel) CPUs; "inductor", "cudagraphs", "onnxrt" on GPU
        "device": "gpu",
        "dtype": "float32",
        "env_name": "benchmark-gpu",
        "env_file": "benchmark-gpu-conda.yaml",
    },
}

code_configs = {
    "cheetah": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "none",  # https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "none",
        "device": "cpu",
        "dtype": "torch.float32",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
}


def render_script(dirname, script, data, verbose=False):
    jinja_env = Environment(loader=FileSystemLoader(dirname))
    template = jinja_env.get_template(script)
    
    rendered_output = template.render(**data)
    if verbose:
        print(rendered_output)
    output_name = script[:-6]  # remove .jinja
    
    with open(output_name, "w") as f:
        f.write(rendered_output)


def install(code_config):
    config = code_configs[code_config]
    env_name = config["env_name"]
    env_file = config["env_file"]
    code = config["code"]

    subprocess.run([f"{conda}", "env", "remove", "-q", "-y", "-n", f"{env_name}"], capture_output=True, check=False)  # ok to fail if does not exist
    subprocess.run([f"{conda}", "env", "create", "-q", "-y", "-f", f"{env_file}"], check=True)

    if code == "impactx":
        data = config.copy()
        data["build_nproc"] = build_nproc
        data["ImpactX_PRECISION"] = "SINGLE"

        render_script("code_impactx", "install.sh.jinja", data)

        command = f"{conda} run -n {env_name} bash install.sh"
        subprocess.run(command, shell=True, check=True)

    elif code == "cheetah":
        command = f"{conda} run -n {env_name} pip install cheetah-accelerator=={config['version']}"
        subprocess.run(command, shell=True, check=True)

    else:
        raise RuntimError(f"Code '{code_config}' not implemented!")


def find_timing_lines(stdout, patterns):
    time_ns = {}
    for line in stdout:
        for key, value in patterns.items():
            match = value.search(line)
            if match:
                time_ns[key] = int(match.group(1))
    return time_ns


def bench(code_config, npart, nruns=5):
    time_ns = None

    config = code_configs[code_config]
    env_name = config["env_name"]

    data = config.copy()
    data["npart"] = npart

    render_script("htu", "run_cheetah_impactx.py.jinja", data)

    time_ns_min = {"track_ns": sys.float_info.max}

    for nrun in range(nruns):

        command = f"{conda} run -n {env_name} python run_cheetah_impactx.py"
        result = subprocess.run(command, shell=True, check=False, capture_output=True)
        stdout = result.stdout.decode("utf-8").split("\n")
        stderr = result.stderr.decode("utf-8")

        print(f"'{code_config}' w/ {npart} particles standard output:", stdout)
        # print("'{code_config}' w/ {npart} particles standard error:", stderr)
        # print("'{code_config}' w/ {npart} particles return code:", result.returncode)

        if result.returncode != 0:
            raise RuntimeError(f"config '{code_config}' w/ {npart} particles failed with: {stderr}")

        patterns = {
            "track_ns": re.compile(r"Track: (.*)ns"),
        }
        time_ns = find_timing_lines(stdout, patterns)

        time_ns_min["track_ns"] = min(time_ns_min["track_ns"], time_ns["track_ns"])

    return time_ns_min



def merge_dicts(original, new_dict):
    """Overwrite existing keys in original and insert new ones"""
    from collections.abc import Mapping
    from copy import deepcopy

    result = deepcopy(original)

    for key, value in new_dict.items():
        if isinstance(value, Mapping):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = deepcopy(new_dict[key])

    return result


def save_timings(timings):
    """Open an existing timings file and update it"""
    import yaml

    timing_file = "timings.yaml"

    # read existing file
    if os.path.exists(timing_file):
        with open(timing_file, "r") as file:
            previous_data = yaml.safe_load(file)
    else:
        previous_data = {}

    # update benchmarks
    merged_timings = merge_dicts(previous_data, timings)

    # save
    with open(timing_file, "w") as file:
        yaml_format = yaml.dump(merged_timings, default_flow_style=False)
        file.write(yaml_format)

    print(yaml_format)


hn = socket.gethostname()
timings = {}

# HTU Benchmark
#
for code_config, _ in code_configs.items():  # TODO: CPU 1-N threads, GPU
    timings[code_config] = {}
    timings[code_config][hn] = {}
    timings[code_config][hn]["config"] = code_configs[code_config]

    # We vary the number of CPU threads, to see if the code benefits from threading (expectation: linear speedup).
    # We compare CPU and GPU runs, to see if the code benefits from GPU acceleration (expectation: ~10x, depending on hardware).
    install(code_config)

    # we vary the number of particles to push in the beam,
    # to see if a code can make efficient use of L1/L2/L3 caches
    for npart in nparts:
        str_npart = str(npart)
        timings[code_config][hn][str_npart] = bench(code_config, npart)
        timings[code_config][hn][str_npart]

        # calculate particle push time
        timings[code_config][hn][str_npart]["push_per_sec"] = npart / timings[code_config][hn][str_npart]["track_ns"] * 1e9

save_timings(timings)
