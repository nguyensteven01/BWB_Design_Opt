# BWB Design Optimization

This repository contains Python scripts for reproducing figures and analyses from our [paper](https://arxiv.org/abs/2604.26134).
Distribution A: Approved for public release; distribution unlimited. Case no. AFRL-2026-2051.

## Main-directory files

- `design_optimization.py` — Runs the blended wing body design optimization problems. The user is prompted to choose between the volume-maximized, direction-maximized, volume-maximized + directionally-constrained, and Gramian-based problems.
- `compare_performance_lq.py` — Compares linear tracking performance for initial and optimized designs.
- `pitch_integrator_nonlinear.py` — Runs nonlinear pitch tracking simulations and plots results and errors.
- `pitch_integrator_nonlinear_compare.py` — Compares nonlinear pitch tracking of the Gramian-optimized design and the reachability-optimized design.
- `flight_performance_metrics.py` — Computes aerodynamic efficiency and power metrics between the reachability-optimized designs and the initial design.
- `plot_aero_data_sweeps.py` — Plots a sweep of the aerodynamic data to show the longitudinal forces as pitch angle and elevator control surface are varied.
- `plot_gramian_eigenvectors.py` — Plots controllability Gramian eigenvectors.
- `process_plot_data.py` — Plots reachable sets for the optimized designs compared with the initial design.

## Dependencies

Use Python 3 and install these packages:

```bash
python -m pip install numpy scipy pandas matplotlib h5py pyslsqp control
```