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

# general configs
#

hostname = "axel-dell"

# experiments (scenarios)
scenarios = ["htu_1000", "htu_10000", "htu_100000", "htu_1000000"]

code_config_colors = {
    "impactx": "tab:red",
    "cheetah": "tab:blue"
}
code_config_hatches = {
    "impactx": ["", "/", "-", "+", "o", ".", "*", "\\", "|"],
    "cheetah": ["", "/", "-", "+", "o", ".", "*", "\\", "|"],
}


def load_timings():
    import yaml

    timing_file = "timings.yaml"
    with open(timing_file, "r") as file:
        timings = yaml.safe_load(file)

    return timings


timings = load_timings()
print(timings)

code_configs = []
for key, value in timings.items():
    code_configs += [key]
print(code_configs)

fig, ax = plt.subplots(layout='constrained')

x = np.arange(len(scenarios))  # the label locations
width = 1 / len(code_configs)  # the width of the bars


multiplier = 0
config_i = {
    "cheetah": 0,
    "impactx": 0,
}
for code_config in code_configs:
    code = timings[code_config][hostname]["config"]["code"]
    
    # irrelevant experiments:
    # - cudagraphs backend in torch is always very slow for Cheetah
    if "cudagraphs" in code_config:
        continue

    if not ("cuda" in code_config or "gpu" in code_config):
        continue
    #if not "cheetah" in code_config:
    #    continue
    #if not "1cpu" in code_config:
    #    continue
    
    color = code_config_colors[code]
    hatch = code_config_hatches[code][config_i[code]]

    measurements = []
    for scenario in scenarios:
        measurements += [timings[code_config][hostname][scenario]["push_per_sec"]]
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        measurements,
        width,
        label=code_config,
        color=color
    )
    for bar in rects:
        bar.set_hatch(hatch)

    ax.bar_label(rects, padding=3)
    multiplier += 1
    config_i[code] += 1

plt.legend()
ax.set_xticks(x + width, scenarios)
ax.set_ylabel("particles / second")

plt.show()

