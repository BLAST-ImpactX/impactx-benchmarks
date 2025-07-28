#!/usr/bin/env python3
#
# Requirements in parent env:
# - conda/mamba/miniconda
# - jinja2
# - git

import re
import subprocess

from jinja2 import Environment, FileSystemLoader


# general configs
#
build_nproc = 12
conda = "mamba"

code_configs = {
    "impactx": {
        "code": "impactx",
        "version": "25.07",
        "gh_owner": "BLAST-ImpactX",
        "CXXFLAGS": "-march=native -ffast-math",
        "ImpactX_COMPUTE": "OMP",
        "ImpactX_SIMD": "OFF",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "impactx-simd": {
        "code": "impactx",
        "version": "topic-simd",
        "gh_owner": "ax3l",
        "CXXFLAGS": "-march=native -ffast-math",
        "ImpactX_COMPUTE": "OMP",
        "ImpactX_SIMD": "ON",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    # TODO: ImpactX CUDA, Cheetah cpu/gpu device
    "cheetah": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "none",  # https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "none",
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    },
    "cheetah-compiled-default": {
        "code": "cheetah",
        "version": "0.7.4",
        "compile_mode": "default",  # TODO: try also "max-autotune" on CPUs https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        "compile_backend": "inductor",  # TODO: try also "ipex" on Intel CPUs
        "env_name": "benchmark-cpu",
        "env_file": "benchmark-cpu-conda.yaml",
    }
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


def bench(code_config, npart):
    time_ns = None

    config = code_configs[code_config]
    env_name = config["env_name"]

    data = config.copy()
    data["npart"] = npart

    render_script("htu", "run_cheetah_impactx.py.jinja", data)

    command = f"{conda} run -n {env_name} python run_cheetah_impactx.py"
    result = subprocess.run(command, shell=True, check=False, capture_output=True)
    stdout = result.stdout.decode("utf-8").split("\n")
    stderr = result.stderr.decode("utf-8")

    print(f"'{code_config}' standard output:", stdout)
    # print("'{code_config}' standard error:", stderr)
    # print("'{code_config}' return code:", result.returncode)

    if result.returncode != 0:
        raise RuntimeError(f"config '{code_config}' failed with: {stderr}")

    patterns = {
        "track_ns": re.compile(r"Track: (.*)ns"),
    }
    time_ns = find_timing_lines(stdout, patterns)

    return time_ns

# HTU Benchmark
#
timings = {}
for code_config, _ in code_configs.items():  # TODO: CPU 1-N threads, GPU

    # We vary the number of CPU threads, to see if the code benefits from threading (expectation: linear speedup).
    # We compare CPU and GPU runs, to see if the code benefits from GPU acceleration (expectation: ~10x, depending on hardware).
    install(code_config)

    # we vary the number of particles to push in the beam,
    # to see if a code can make efficient use of L1/L2/L3 caches
    for npart in [1_000, 10_000, 100_000]:  # TODO: add , 1_000_000, 10_000_000]:
        str_npart = str(npart)
        timings[str_npart] = {}
        timings[str_npart][code_config] = bench(code_config, npart)

        # calculate particle push time
        timings[str_npart][code_config]["push_per_sec"] = npart / timings[str_npart][code_config]["track_ns"] * 1e9

# TODO: proper storage and plotting
print(timings)
print(timings["100000"]["cheetah"]["push_per_sec"] / timings["100000"]["impactx-simd"]["push_per_sec"])
#print(timings)
