# Reachability-Based Design Optimization for a Blended Wing Body Aircraft

This repository contains the Python codebase accompanying a paper we have submitted to AIAA Journal. The code implements a reachability-based framework for optimizing the longitudinal design parameters of a Blended Wing Body (BWB) aircraft, using control-theoretic tools including LQR tracking, reachable set computation, and nonlinear simulation.

Distribution A: Approved for public release; distribution unlimited. Case no. AFRL-2026-1943.

---

## Repository Structure

```
├── Aero_data/
│   └── aero_7dim2.npz          # Tabulated aerodynamic data (7-dimensional grid)
├── Output/
│   └── aero_7dim_asym/
│       ├── N6_initial/         # Reachability data for the initial design
│       ├── N6_volmax/          # Results: volume-maximizing optimization
│       ├── N6_110max/          # Results: directional-extent maximizing optimization
│       └── N6_volmax_110con/   # Results: volume maximization with directional constraint
├── interpolatory_model.py
├── nonlinear_model.py
├── trim_aircraft.py
├── solve_rde.py
├── ref_trackers.py
├── compute_inf_reach_set.py
├── reach_set_functions.py
├── design_optimization.py
├── process_plot_data.py
├── compare_performance_lq.py
├── compare_performance_integral_lqr.py
├── pitch_integrator_nonlinear.py
├── plot_aero_data_sweeps.py
├── plotters.py
├── read_hdf5.py
└── progress_bar.py
```

---
## Reproducing Paper Results

| Result | Script | Notes |
|--------|--------|-------|
| Figure 3 | `plot_aero_data_sweeps.py` | Run directly; requires `Aero_data/aero_7dim2.npz` |
| Geometry Optimizations | `design_optimization.py` | Optimized variables are saved to HDF5 files in `Output/aero_7dim_asym/...` |
| Figures 5–7 | `process_plot_data.py` | CSV files containing reachable set data for plotting should already be included. Optionally setting `proj` to `True` will add the projection visualization. |
| Figures 8–10 | `compare_performance_lq.py` | HDF5 outputs from `design_optimization.py` should already be included. Error data will be printed to terminal. |
| Figures 11–13 | `compare_performance_integral_lqr.py` | HDF5 outputs from `design_optimization.py` should already be included. Error data will be printed to terminal. |
| Figures 14–16 | `pitch_integrator_nonlinear.py` | HDF5 outputs from `design_optimization.py` should already be included. Error data will be printed to terminal. |
---

## File Descriptions

### Core Model Files

**`interpolatory_model.py`**  
Loads the tabulated aerodynamic data from the `.npz` file and builds `RegularGridInterpolator` objects for all aerodynamic force and moment derivatives (e.g., $X_v$, $Z_\alpha$, $M_{\delta_e}$). Also provides `build_AB()`, which evaluates these interpolators at a given flight point to construct the linearized longitudinal state-space matrices `A` and `B`. The state vector is $[V,\ \alpha,\ q,\ \theta]^\top$ (airspeed, angle of attack, pitch rate, pitch angle) and inputs are $[\delta_{th},\ \delta_e]$ (thrust throttle, elevator deflection).

**`trim_aircraft.py`**  
Computes the trimmed flight condition (angle of attack, elevator deflection, thrust, and pitch angle) for a given airspeed, climb angle, center halfspan, and wing halfspan using `scipy.optimize`. Trim is found by solving the nonlinear equations of motion as equality constraints with a penalty-based optimizer.

### Control Design Files

**`solve_rde.py`**  
Solves the Matrix Riccati Differential Equation (RDE) backward in time using `scipy.integrate.solve_ivp`. Optionally solves the coupled co-state ODE for optimal reference tracking. Returns time-indexed arrays of $P(t)$ and $\eta(t)$ for use in finite-horizon LQR controllers.

**`ref_trackers.py`**  
Implements two linear reference tracking strategies for the longitudinal system:
- `lqr_ref_track()`: Finite-horizon LQR tracking that solves the RDE backward in time and simulates the optimal closed-loop. Supports velocity, pitch, or angle-of-attack tracking.
- `integral_ref_track()`: Infinite-horizon LQR with an augmented integrator state to eliminate steady-state tracking error, with anti-windup logic.

**`nonlinear_model.py`**  
Implements the full nonlinear longitudinal equations of motion for the BWB. Provides `nonlinear_track_velocity()` and `nonlinear_track_pitch()`, which simulate the aircraft under an LQR+integrator controller designed around the linearized model. Supports a switching flight condition (two consecutive trim points connected at a specified `switch_time`). Used to evaluate controlled performance of nonlinear dynamics.

### Reachability Analysis Files

**`compute_inf_reach_set.py`**  
Computes the reachable set of a continuous-time LTI system under bang-bang (infinity-norm optimal) control over a finite time horizon. Samples costate initial conditions on the unit sphere in $\mathbb{R}^4$, simulates the resulting state trajectory for each, and returns the array of final states and associated control costs. Optionally saves results to CSV.

**`reach_set_functions.py`**  
Utility functions for working with the reachable set data:
- `compute_hull()`: Computes the convex hull of reachable state samples.
- `plot_hull_projection()`: Projects a 4D convex hull onto a 2D plane.
- `unpack_reach_set_data()`: Reads a CSV output from `compute_inf_reach_set.py` and reshapes the data for analysis.

### Design Optimization File

**`design_optimization.py`**  
The main optimization driver. The design variables are center halfspan (`chs`) and wing halfspan (`whs`) of the BWB. At each function evaluation, this script re-trims the aircraft, re-builds `A` and `B`, computes the reachable set, and evaluates the objective/constraints. Three optimization problems are solved using `pyslsqp`:
1. **Volume maximization** (`N6_volmax`): Maximize the volume of the convex hull of the reachable set.
2. **Directional extent maximization** (`N6_110max`): Maximize the projected extent of the reachable set along a specified direction in the pitch rate–pitch angle plane (110° direction).
3. **Volume maximization with directional constraint** (`N6_volmax_110con`): Maximize volume subject to a minimum 15% increase in directional extent.

### Plotting and Post-Processing Files

**`plot_aero_data_sweeps.py`** → **Figure 3**  
Generates 2D color mesh plots of aerodynamic force/moment data (e.g., axial force $F_x$) as functions of two selected parameters (e.g., elevator deflection and pitch angle), with all other parameters held fixed. Used to visualize the aerodynamic database sweeps.

**`design_optimization.py`** → **Figure 4**  
The optimization convergence plots are saved automatically during the `pyslsqp` runs via the `save_figname` argument. This plot also generates HDF5 files which contain the optimal values of center halfspan and wing halfspan for each of the optimization problems.

**`process_plot_data.py`** → **Figures 5, 6, 7**  
Loads reachable set data from CSV for both the initial and optimized designs and generates 2D projections of the convex hulls. Figures show projections onto the $(q,\ \theta)$ plane and the $(V,\ \alpha)$ plane, with optional overlay of the directional projection line. Quantitative comparisons (volume change, extent change) are printed to the terminal.

**`compare_performance_lq.py`** → **Figures 8, 9, 10**  
Compares the finite-horizon LQR tracking performance (`lqr_ref_track`) between the initial and each of the three optimized designs for pitch angle and airspeed tracking tasks. For each model and case, generates a two-panel plot: (1) tracked state vs. reference, and (2) control history. Tracking error and control effort L2 norms are printed to the terminal.

**`compare_performance_integral_lqr.py`** → **Figures 11, 12, 13**  
Analogous to `compare_performance_lq.py` but using the integral LQR controller (`integral_ref_track`). Evaluates both pitch and velocity tracking for the initial and optimized designs, producing the same two-panel format. Tracking error and control effort L2 norms are printed to the terminal.

**`pitch_integrator_nonlinear.py`** → **Figures 14, 15, 16**  
Runs nonlinear closed-loop simulations for the pitch-angle tracking task across the initial and all three optimized designs. The aircraft is commanded to follow a two-segment trajectory (switching flight condition at $t=20$ sec). Calls `error_bars()` to generate a grouped bar chart of relative tracking error improvements, and `niceplot_one_at_a_time()` to generate detailed trajectory and control comparisons for each optimized model vs. the initial.

### Utility Files

**`plotters.py`**  
Contains `niceplot_one_at_a_time()` for generating plots for nonlinear simulations (airspeed, AOA, pitch rate, pitch angle, controls) comparing an initial and an optimized model, and `error_bars()` for computing and plotting relative tracking error reductions across all model types.

**`read_hdf5.py`**  
Utilities for reading optimization results stored in HDF5 format by `pyslsqp`. Provides `print_hdf5_contents()`, `print_results_group()`, and `extract_dataset_from_results()` for extracting specific datasets (e.g., optimized design variables `x`) from saved output files.

**`progress_bar.py`**  
A simple terminal progress bar used during the reachable set sampling loop in `compute_inf_reach_set.py`.

---

## Dependencies

```
numpy
scipy
matplotlib
pandas
control          # python-control
pyslsqp
h5py
```