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
    #    return True
    if not "1cpu" in config_name:
        return True

    return False

# experiments (scenarios)
scenarios = ["htu_1000", "htu_10000", "htu_100000", "htu_1000000"]

# coloring & hatching of bar plots
code_config_colors = {
    "impactx": "tab:red",
    "cheetah": "tab:blue"
}
code_config_hatches = {
    "impactx": ["", "///", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
    "cheetah": ["///", "", "---", "+++", "ooo", "...", "***", "\\\\\\", "|||"],
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
    code_configs += [key]

fig, ax = plt.subplots(layout='constrained')

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
    rects = ax.bar(
        x + offset,
        measurements,
        width,
        label=code_config,
        color=color,
        edgecolor='black',
    )
    for bar in rects:
        bar.set_hatch(hatch)

    str_speedup_over_baseline = [f'{val:.1f}x' for val in speedup_over_baseline]
    ax.bar_label(rects, labels=str_speedup_over_baseline, padding=3)
    bar_x_offset_multiplier += 1
    config_i[code] += 1

plt.legend()
ax.set_xticks(
    x + width,
    scenarios
)
ax.set_ylabel("particles / second")

plt.show()

