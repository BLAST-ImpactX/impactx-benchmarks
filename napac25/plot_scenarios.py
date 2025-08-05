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

import matplotlib.pyplot as plt
import numpy as np


# general configs #############################################################
#

# filtering of configs
hostname = "axel-dell"

def filter_config(config_name, code_config_vals):
    code = code_config_vals["code"]

    if code == "cheetah":
        # irrelevant/too detailed experiments:
        # - cudagraphs backend in torch is always very slow
        if "cudagraphs" in config_name:
            return True
        # - inductor and inductor-fm are the same (always fast-math)
        if "inductor" in config_name and not "-fm" in config_name:
            return True
    elif code == "impactx":
        if "cpu" in config_name:
            # irrelevant/too detailed experiments:
            # - always show fm-simd as in Cheetah
            if "-fm" in config_name and not "-simd" in config_name:
                return True
            if "-simd" in config_name and not "-fm" in config_name:
                return True
            if "-autovec" in config_name:  # no fm and no simd, odd comparison
                return True

    #if not ("cuda" in config_name or "gpu" in config_name):
    #   return True
    #if not "1cpu" in config_name:
    #    return True
    if "6cpu" in config_name:
        return True

    #if "dp" in config_name:
    #    return True
    if not "dp" in config_name:
        return True

    # does not fit on Axel's laptop, enable on perlmutter
    if config_name == "cheetah-cuda-dp":
        return True

    return False

# experiments (scenarios)
#scenarios = ["htu_1000", "htu_10000", "htu_100000", "htu_1000000"]
#scenario_labels = ["1k", "10k", "100k", "1M"]
#scenarios = ["htu_10000", "htu_100000", "htu_1000000"]
#scenario_labels = ["10k", "100k", "1M"]

#scenarios = ["spacecharge_1000", "spacecharge_10000", "spacecharge_100000", "spacecharge_1000000"]
#scenario_labels = ["1k", "10k", "100k", "1M"]
scenarios = ["spacecharge_1000000"]
scenario_labels = ["1M"]

# coloring & hatching of bar plots
code_config_colors = {
    "impactx": "tab:red",
    "cheetah": "tab:blue"
}
code_config_hatches = {
    "impactx": ["", "///", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
    "cheetah": ["///", "", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
}
# CPU-only or GPU-only plots
code_config_labels = {
    "cheetah-1cpu": "Cheetah",
    "cheetah-1cpu-dp": "Cheetah",
    "cheetah-cuda": "Cheetah",
    "cheetah-cuda-dp": "Cheetah",
    "cheetah-1cpu-inductor-fm-simd": "Cheetah: compiled",
    "cheetah-1cpu-inductor-fm-simd-dp": "Cheetah: compiled",
    "cheetah-cuda-inductor-fm": "Cheetah: compiled",
    "cheetah-cuda-inductor-fm-dp": "Cheetah: compiled",
    "impactx-1cpu-fm-simd": "ImpactX",
    "impactx-1cpu-fm-simd-dp": "ImpactX",
    "impactx-cuda-fm": "ImpactX",
    "impactx-cuda-fm-dp": "ImpactX",
}
# CPU + GPU plot
code_config_labels = {
    "cheetah-1cpu-dp": "CPU",
    "cheetah-cuda-dp": "GPU",
    "cheetah-1cpu-inductor-fm-simd-dp": "CPU: compiled",
    "cheetah-cuda-inductor-fm-dp": "GPU: compiled",
    "impactx-1cpu-fm-simd-dp": "CPU",
    "impactx-cuda-fm-dp": "GPU",
}

###############################################################################

# data handling and plotting ##################################################
#

def load_timings():
    import yaml

    timing_file = "timings.yaml"
    with open(timing_file, "r") as file:
        timings = yaml.safe_load(file)

    return timings


timings = load_timings()

code_configs = []
for key, value in timings.items():
    if hostname in timings[key]:
        code_configs += [key]

fig, ax = plt.subplots(figsize=(4.5, 2.2), layout='constrained')

x = np.arange(len(scenarios))  # the label locations

# num_shown_configs = len(code_configs)  # if all are plotted
num_shown_configs = sum([
    0 if filter_config(
            code_config,
            timings[code_config][hostname]["config"]
    ) else 1 for code_config in code_configs
])
width = 1 / (num_shown_configs + 1)  # the width of the bars
bar_x_offset_multiplier = 0  # -num_shown_configs / 2

config_i = {
    "cheetah": 0,
    "impactx": 0,
}
push_per_sec_baseline = []  # first config is baseline
for code_config in code_configs:
    if filter_config(code_config, timings[code_config][hostname]["config"]):
        continue

    code = timings[code_config][hostname]["config"]["code"]

    color = code_config_colors[code]
    hatch = code_config_hatches[code][config_i[code]]

    if len(push_per_sec_baseline) == 0:
        for scenario in scenarios:
            push_per_sec_baseline += [timings[code_config][hostname][scenario]["push_per_sec"]]

    measurements = []
    speedup_over_baseline = []
    for si, scenario in enumerate(scenarios):
        measurements += [timings[code_config][hostname][scenario]["push_per_sec"]]
        speedup_over_baseline += [
            timings[code_config][hostname][scenario]["push_per_sec"] /
            push_per_sec_baseline[si]
        ]
    offset = width * bar_x_offset_multiplier
    label = code_config_labels[code_config] if code_config in code_config_labels else code_config
    rects = ax.bar(
        x + offset,
        measurements,
        width,
        #label=code_config,
        label=label,
        color=color,
        edgecolor='black',
    )
    for bar in rects:
        bar.set_hatch(hatch)

    str_speedup_over_baseline = [f'{val:.1f}' for val in speedup_over_baseline]
    ax.bar_label(rects, labels=str_speedup_over_baseline, padding=3)
    bar_x_offset_multiplier += 1
    config_i[code] += 1

ymin_data, ymax_data = ax.get_ylim()
ax.set_ylim(ymin_data, ymax_data * 1.15)

ax.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncols=num_shown_configs if num_shown_configs < 3 else 3,
    mode="expand",
    borderaxespad=0.0,
)
ax.set_xticks(
    x + width,
    #scenarios
    scenario_labels
)
ax.set_xlabel("particles / beam")
ax.set_ylabel("particles / second")

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
offset_text = ax.yaxis.get_offset_text()
# Set a new position for the offset text (e.g., slightly to the right and up)
# Coordinates are in axes coordinates (0 to 1)
offset_text.set_position((-0.1, 1.0))

plt.show()

