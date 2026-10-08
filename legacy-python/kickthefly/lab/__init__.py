"""Lab mode: everything that treats the simulation as an experiment rather than a toy.

  assays      standard assays (T-maze conditioning, looming escape, sugar dose-response)
  benchmark   simulation throughput measurement
  challenges  Play mode's three challenges, each built on one of those assays
  headless    the no-window entry point for --validate, --protocol, --benchmark and audits
  lab         the Lab screens: hub, parameters, assumptions, validation, export, protocols
  labjobs     running assays over many seeds in worker processes, with same-seed controls
  labstats    confidence intervals and significance tests
  protocol    YAML experiment files
  recorder    live spike and rate recording to CSV/npz (and NWB, via nwbexport)
  validation  which published results this simulation reproduces, with pass criteria fixed beforehand
"""
