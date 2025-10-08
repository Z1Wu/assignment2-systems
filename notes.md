uv run nsys profile -t cuda --stats=true -o profile_results/result python cs336_systems/benchmark.py


uv run nsys profile --stats=true -o profile_results/result_bp_optimize python cs336_systems/benchmark.py
