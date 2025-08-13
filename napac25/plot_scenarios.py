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
#hostname = "axel-dell"
hostname = "perlmutter"

napac_figure = 3

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

    if napac_figure == 1:
        if not "1cpu" in config_name:
            return True
        if "dp" in config_name:
            return True

    if napac_figure == 2:
        if not ("cuda" in config_name or "gpu" in config_name):
            return True
        if "dp" in config_name:
            return True
            
    if napac_figure == 3:
        if not "1cpu" in config_name:
            return True
        if "6cpu" in config_name:
            return True
        if not "dp" in config_name:
            return True

    #if not "1cpu" in config_name:
    #    return True
    #if not "6cpu" in config_name:
    #    return True
    #if "cpu" in config_name:
    #    return True

    #if "6cpu" in config_name:
    #    return True

    #if "dp" in config_name:
    #    return True
    #if not "dp" in config_name:
    #    return True

    # does not fit on Axel's laptop, enable on perlmutter
    #if config_name == "cheetah-cuda-dp":
    #    return True

    return False

# experiments (scenarios)
if napac_figure == 1:
    scenarios = ["htu_1000", "htu_10000", "htu_100000", "htu_1000000"]
    scenario_labels = ["1k", "10k", "100k", "1M"]
if napac_figure == 2:
    scenarios = ["htu_10000", "htu_100000", "htu_1000000", "htu_10000000"]
    scenario_labels = ["10k", "100k", "1M", "10M"]

#scenarios = ["spacecharge_1000", "spacecharge_10000", "spacecharge_100000", "spacecharge_1000000"]
#scenario_labels = ["1k", "10k", "100k", "1M"]
if napac_figure == 3:
    scenarios = ["spacecharge_1000000", "spacecharge_1000000"]
    scenario_labels = ["1M", "1M"]

# coloring & hatching of bar plots
code_config_colors = {
    "cpu": {
        "impactx": "tab:orange",
        "cheetah": "tab:cyan"
    },
    "cuda": {
        "impactx": "tab:red",
        "cheetah": "tab:blue"
    },
}
if napac_figure == 3:
    code_config_hatches = {
        "impactx": ["", ""],
        "cheetah": ["///", "", "\\\\\\", ""]
    }
else:
    code_config_hatches = {
        "impactx": ["", "///", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
        "cheetah": ["///", "", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
    }
if napac_figure == 1 or napac_figure == 2:
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
if napac_figure == 3:
    # CPU + GPU plot
    code_config_labels = {
        "cheetah-1cpu-dp": "Cheetah CPU",
        "cheetah-cuda-dp": "Cheetah GPU",
        "cheetah-1cpu-inductor-fm-simd-dp": "Cheetah CPU: compiled",
        "cheetah-cuda-inductor-fm-dp": "Cheetah GPU: compiled",
        "impactx-1cpu-fm-simd-dp": "ImpactX CPU",
        "impactx-cuda-fm-dp": "ImpactX GPU",
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

fig = plt.figure(figsize=(4.7, 2.2))  #, layout='constrained')
ax = fig.gca()

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

    if "cpu" in code_config:
        color = code_config_colors["cpu"][code]
    elif "cuda" in code_config:
        color = code_config_colors["cuda"][code]
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

    if napac_figure == 3:
        str_speedup_over_baseline = [f'{val:.1f}x' for val in speedup_over_baseline]
    else:
        str_speedup_over_baseline = [f'{val:.2g}x' for val in speedup_over_baseline]
    ax.bar_label(rects,
        labels=str_speedup_over_baseline,
        padding=0,
        fontsize=9,
        rotation=0)
    bar_x_offset_multiplier += 1
    config_i[code] += 1

ymin_data, ymax_data = ax.get_ylim()
if napac_figure == 3:
    #ax.set_yscale('log')
    ax.set_ylim(ymin_data, ymax_data * 1.5)
else:
    ax.set_ylim(ymin_data, ymax_data * 1.15)

# fix weird ordering columns/rows
handles, labels = ax.get_legend_handles_labels()
#if napac_figure == 3:
#    new_order = [0, 2, 4, 1, 3, 5]
#    reordered_handles = [handles[i] for i in new_order]
#    reordered_labels = [labels[i] for i in new_order]
#    handles, labels = reordered_handles, reordered_labels

ax.legend(
    handles,
    labels,
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncols=num_shown_configs if num_shown_configs <= 3 else 2,
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

#if napac_figure != 3:
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
offset_text = ax.yaxis.get_offset_text()
# Set a new position for the offset text (e.g., slightly to the right and up)
# Coordinates are in axes coordinates (0 to 1)
offset_text.set_position((-0.1, 1.0))

plt.tight_layout()
plt.show()

