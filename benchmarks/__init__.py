"""impactx-benchmarks: automated correctness & performance benchmarks of beam-dynamics codes.

The package is split into small, single-responsibility modules:

* :mod:`benchmarks.registry`  -- codes, scenarios, configs and the capability matrix
* :mod:`benchmarks.render`    -- Jinja rendering of per-(code, scenario) run scripts
* :mod:`benchmarks.metadata`  -- host / OS / CPU / compiler / version capture
* :mod:`benchmarks.results`   -- results schema, incremental (de)serialization
* :mod:`benchmarks.validate`  -- physics-correctness classification
* :mod:`benchmarks.runner`    -- orchestration loop (build envs, run, collect, save)
* :mod:`benchmarks.plotting`  -- status/physics-aware bar charts
* :mod:`benchmarks.build`     -- build/install the from-source codes via their pixi envs
* :mod:`benchmarks.publish`   -- commit results+plots to the ``benchmarks`` branch
"""

__version__ = "0.1.0"
