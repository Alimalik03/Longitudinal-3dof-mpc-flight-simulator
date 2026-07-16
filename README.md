# Longitudinal 3DOF Flight Simulator with Explicit MPC

A closed-loop longitudinal (3‑degree‑of‑freedom) aircraft flight simulator driven by an **explicit Model Predictive Controller (eMPC)**, with a custom active-set quadratic programming solver for input-constraint handling, WGS84 geodetic conversion for trajectory generation, and live visualization through **FlightGear**.

The simulator models altitude, pitch, airspeed, angle of attack, and pitch rate dynamics of a fixed-wing aircraft, tracks altitude/airspeed setpoints using a receding-horizon eMPC law, enforces elevator deflection and rate limits via a constrained QP correction step, and streams the resulting trajectory to FlightGear for real-time 3D playback.

## Features

- **Linear state-space aircraft model** (7 states, 2 inputs, 5 outputs) with continuous-to-discrete (zero-order hold) conversion
- **Explicit MPC controller** using eigen-decomposition-based prediction (F/G) matrices and an unconstrained analytic gain
- **Custom constrained QP solver** (multiplicative iterative update, active-set style) to project the control input back onto the feasible set when constraints are violated
- **WGS84 NED ↔ LLA coordinate conversion** for translating the simulated trajectory into geodetic coordinates
- **FlightGear real-time visualization** via UDP FDM streaming
- **Performance dashboard**: tracking plots, RMS error metrics, and a console summary report

## Project Structure

| File | Description |
|---|---|
| `Aircraft_model.py` | Aircraft state-space model, discretization, state propagation, logging, and result plotting |
| `Longitudinal_3DOF_sim.py` | Main simulation driver — sets up the model/controller, runs the closed loop, and streams to FlightGear |
| `MPC_script.py` | Explicit MPC controller design: prediction (F/G) matrices, gain, and constraint definitions |
| `quadratic_optim_solver.py` | Custom quadratic programming solver used for constrained input correction |
| `metric_plot.py` | Performance metrics (RMS tracking error, etc.) and dashboard plotting |
| `WGS84.py` | WGS84 ellipsoid coordinate transformations (LLA ↔ NED) |

## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib`
- `control`
- `pandas`
- `flightgear_python` (only required for the live FlightGear visualization)
- [FlightGear](https://www.flightgear.org/) (optional, for real-time playback)

Install dependencies:

```bash
pip install numpy scipy matplotlib control pandas flightgear_python
```

## Usage

Run the full pipeline (model setup → controller design → closed-loop simulation → metrics → FlightGear streaming):

```bash
python Longitudinal_3DOF_sim.py
```

This will:
1. Build and discretize the longitudinal state-space aircraft model
2. Design the eMPC controller (prediction matrices + gain) and set elevator/elevator-rate constraints
3. Run the closed-loop simulation over the configured time horizon, applying constrained MPC control at each step
4. Print a performance summary and display a tracking/metrics dashboard
5. Stream the resulting trajectory to FlightGear (requires FlightGear running and listening on the configured ports)

### Running without FlightGear

If you don't have FlightGear installed, comment out or remove the `self.start_flightgear()` call at the end of `LongitudinalControl.run()` in `Longitudinal_3DOF_sim.py` — the simulation, metrics, and plots will still run normally.

### FlightGear setup

The simulator expects FlightGear to be listening for FDM data on UDP port `5501` (receive) and `5502` (transmit) on `localhost`. Launch FlightGear with the appropriate `--native-fdm` socket options before running the script, or adjust the ports/host in `start_flightgear()` to match your setup.

## Model Overview

**State vector** (relative to trim): `[Δh, Δθ, Δu, Δα, Δq, δe_cmd, δe]` — altitude, pitch angle, airspeed, angle of attack, pitch rate, commanded elevator, and actual elevator deflection.

**Inputs**: throttle (`δt`) and elevator rate command.

**Outputs**: altitude, pitch angle, airspeed, angle of attack, and pitch rate — split into a slow-sampling group (altitude, pitch) and a fast-sampling group (airspeed, AoA, pitch rate) for the eMPC prediction horizons.

The controller solves an unconstrained least-squares MPC law analytically, then checks the elevator deflection and rate constraints; if violated, a constrained QP correction (`PQP` in `quadratic_optim_solver.py`) recomputes a feasible input.

HAVE A SMOOTH FLIGHT !!!
