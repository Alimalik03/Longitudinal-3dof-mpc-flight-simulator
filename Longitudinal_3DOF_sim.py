from MPC_script import eMPC
from quadratic_optim_solver import PQP
from Aircraft_model import System
from WGS84 import NED2LLA
import numpy as np
import matplotlib.pyplot as plt
from metric_plot import show_metrics, print_summary
import copy
import time

d2r = np.deg2rad


# ==============================================================================
# LongitudinalControl Class
# ==============================================================================

class LongitudinalControl:
    """
    Longitudinal dynamics-control simulation using Model Predictive Control (MPC).
    """

    def __init__(self):
        """Initialising all subsystems, parameters, and storage containers."""

        # --- Core subsystems ---
        self.system     = System()
        self.controller = None          # populated in setup_controller()

        # --- Simulation parameters ---
        self.dT   = 0.01   # timestep [s]
        self.Time = 60     # total simulation time [s]

        # Derived time indices (set in setup_time())
        self.k_f  = None
        self.dk_1 = None
        self.dk_2 = None
        self.limit = None

        # Prediction horizons [s]
        self.N_p = [3.5, 3.5]

        # --- FlightGear playback state ---
        self.fg_index      = 0
        self.lat0_target   = d2r(34.44417)
        self.lon0_target   = d2r(126.44125)
        self.alt0          = 7           # ground elevation [m]
        self.psi           = d2r(340)    # heading [rad]

        # Trajectory buffers fed to the FlightGear callback
        self.latValues      = []
        self.lonValues      = []
        self.altValues      = []
        self.thetaValues    = []
        self.alphaValues    = []
        self.thetadotValues = []
        self.elevatorValues = []

    # --------------------------------------------------------------------------
    # Setup helpers
    # --------------------------------------------------------------------------

    def setup_system(self):
        """Defining continuous state-space matrices and initial conditions."""

        A_c = np.array([
            [0,      500.0000,  0,       -500.0000,  0,       0,       0      ],
            [0,      0,         0,        0,          1,       0,       0      ],
            [0.0001, -32.1700, -0.0130,  -2.9483,   -1.0283,  0.0016,  0.1018 ],
            [0,      0,        -0.0003,  -0.7506,    0.9281, -0.0000, -0.0016 ],
            [0,      0,         0,       -1.8365,   -1.0271,  0,      -0.1335 ],
            [0,      0,         0,        0,          0,      -1,       0      ],
            [0,      0,         0,        0,          0,       0,      -20.2  ],
        ])

        B_c = np.array([
            [0,    0   ],
            [0,    0   ],
            [0,    0   ],
            [0,    0   ],
            [0,    0   ],
            [1,    0   ],
            [0,    20.2],
        ])

        C_c = np.array([
            [1,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  0,  0],
            [0,  0, -7,  0,  0,  0,  0],
            [0,  0,  0,  1,  0,  0,  0],
            [0,  0,  0,  0,  1,  0,  0],
        ])

        # Initial state vector  [h, theta, u, alpha, q, delta_e_cmd, delta_e]
        XO = np.array([10000, 0.0638, 500, 0.0638, 0, 0.06, -0.0393])

        self.system.updateContinuousStateModel(A_c, B_c, C_c, XO)
        self.C_c = C_c  # store so setup_controller() can access it

        # Discretizing and setting initial control input
        self.system.discretize(self.dT)
        U = np.zeros(self.system.m)
        self.system.updateInitialConditions(U)

        # MPC weight matrices
        Q = np.diag([1, 1, 49, 1, 1])
        R = np.diag([20, 50])
        self.system.updateMPCParameters(Q, R)

    def setup_controller(self):
        """Build the eMPC controller: F/G matrices, constraints, and gain."""

        n, m, p = self.system.n, self.system.m, self.system.p
        self.controller = eMPC(self.N_p, n, m, p)

        # Output sub-matrices
        C_c  = self.C_c          # full output matrix stored by System
        C_1  = C_c[:2]                  # slow sampling rate outputs  (altitude, pitch angle)
        C_2  = C_c[2:]                  # fast sampling rate outputs  (speed, AoA, pitch rate)

        # Prediction / control matrices for each output 
        F_1, G_1 = self.controller.calculateFGMatrices(
            self.system.Ac, self.system.Bc, C_1, self.N_p[0], n)
        F_2, G_2 = self.controller.calculateFGMatrices(
            self.system.Ac, self.system.Bc, C_2, self.N_p[1], n)

        F = np.asarray(np.concatenate((F_1, F_2)))
        G = np.asarray(np.concatenate((G_1, G_2)))
        self.controller.assignFGMatrices(F, G)

        # Input constraints:  elevator rate ∈ [-1, 1],  elevator ∈ [-25°, 25°]
        M_con = np.array([[ 1,  0],
                           [-1,  0],
                           [ 0,  1],
                           [ 0, -1]])
        g_con = np.array([1, 0, 0.4363, 0.4363])
        self.controller.setConstraints(M_con, g_con)

        # Computing optimal feedback gain
        self.controller.calculateGain(self.system.Q, self.system.R)

    def setup_time(self):
        """Pre-compute time indices and allocate logger."""

        self.k_f   = int(self.Time / self.dT)
        self.dk_1  = int(self.N_p[0] / self.dT)
        self.dk_2  = int(self.N_p[1] / self.dT)
        self.limit = int(self.k_f - max(self.dk_1, self.dk_2))
        self.system.prepareLogger(self.k_f)

    def _init_fg_buffers(self):
        """Seed FlightGear trajectory buffers with the initial state."""

        XO = self.system.XO
        self.latValues      = [self.lat0_target]
        self.lonValues      = [self.lon0_target]
        self.altValues      = [XO[0]]
        self.thetaValues    = [XO[1]]
        self.alphaValues    = [XO[3]]
        self.thetadotValues = [XO[4]]
        self.elevatorValues = [XO[6]]

    # --------------------------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------------------------

    def run_simulation(self):
        """
        Execute the MPC closed-loop simulation for `self.limit` timesteps.

        At each step:
          1. Advance the plant one timestep.
          2. Predict future outputs.
          3. Compute unconstrained optimal input.
          4. Project onto constraint set if required (active-set QP).
          5. Log states / inputs and append FlightGear trajectory data.
        """

        # Reference trajectory
        R_bar = self.system.Setpoint_Assignment4(0, self.k_f, self.system.p)

        self._init_fg_buffers()

        x_pos = 0.0   # NED north displacement [m]
        y_pos = 0.0   # NED east  displacement [m]

        for k in range(self.limit):

            # 1. Save previous input, advance plant
            U_old = copy.deepcopy(self.system.U)
            self.system.stepsim()

            # 2. Predicted outputs at horizon endpoints
            Y_pred = np.squeeze(self.controller.F @ self.system.X)

            # 3. Future tracking error
            self.system.E[0] = R_bar[0][k + self.dk_1] - Y_pred[0].item()
            self.system.E[2] = R_bar[1][k + self.dk_2] - Y_pred[2].item()

            # 4. Unconstrained MPC input
            self.system.U = self.controller.K_eMPC @ self.system.E

            # 5. Constrained correction (active-set QP)
            if not self.controller.constraintsSatisfied(
                    self.system.U, self.controller.M_con, self.controller.g_con):

                self.system.f = (
                    -self.controller.G.transpose()
                    @ self.system.Q
                    @ self.system.E
                )
                qp = PQP(self.controller.M_con,
                         self.controller.H_inv,
                         self.controller.g_con,
                         self.system.f)
                qp.optimize()
                self.system.U = -self.controller.H_inv @ (
                    self.system.f
                    + 0.5 * self.controller.M_con.transpose() @ qp.lam
                )

            # 6. Log
            dU = U_old - self.system.U
            self.system.logStatesAndInputs(k, dU)

            # 7. Propagate NED position and append to FlightGear buffers
            curr_speed = self.system.Xlogged[2, k].item()
            curr_theta = self.system.Xlogged[1, k].item()
            curr_D     = self.alt0 - self.system.Xlogged[0, k].item()

            x_pos += self.dT * (curr_speed * np.cos(self.psi) * np.cos(curr_theta))
            y_pos += self.dT * (curr_speed * np.sin(self.psi) * np.cos(curr_theta))

            P_LLA = NED2LLA(x_pos, y_pos, curr_D,
                             self.lat0_target, self.lon0_target, self.alt0)

            self.latValues.append(P_LLA[0])
            self.lonValues.append(P_LLA[1])
            self.altValues.append(self.system.Xlogged[0, k].item())
            self.thetaValues.append(self.system.Xlogged[1, k].item())
            self.alphaValues.append(self.system.Xlogged[3, k].item())
            self.thetadotValues.append(self.system.Xlogged[4, k].item())
            self.elevatorValues.append(self.system.Ulogged[1, k].item())

        # Store final step count and reference for plotting
        self._k_final = k
        self._R_bar   = R_bar

    # --------------------------------------------------------------------------
    # Post-processing
    # --------------------------------------------------------------------------

    def plot_results(self):
        """Generate time-history plots via the System helper."""

        TIME  = np.linspace(0, self._k_final * self.dT, self._k_final)
        self.system.plotResults(TIME, self._R_bar[0], self._R_bar[1])
        plt.show()

    # --------------------------------------------------------------------------
    # FlightGear interface
    # --------------------------------------------------------------------------

    def _fdm_callback(self, fdm_data, event_pipe):
        """
        FlightGear FDM callback.

        Injects pre-computed trajectory data into the FDM at each frame.
        Loops back to the start when the buffer is exhausted.
        """
        i = self.fg_index

        fdm_data.lat_rad              = self.latValues[i]
        fdm_data.lon_rad              = self.lonValues[i]
        fdm_data.alt_m                = self.altValues[i]
        fdm_data.phi_rad              = 0.0
        fdm_data.theta_rad            = self.thetaValues[i]
        fdm_data.psi_rad              = self.psi
        fdm_data.elevator             = self.elevatorValues[i]
        fdm_data.alpha_rad            = self.alphaValues[i]
        fdm_data.thetadot_rad_per_s   = self.thetadotValues[i]

        self.fg_index = (i + 1) % max(len(self.latValues) - 1, 1)
        print(f"FG playback: {self.fg_index / self.limit * 100:.1f} %")

        return fdm_data

    def start_flightgear(self):
        """Connect to FlightGear and stream trajectory data in real-time."""
        from flightgear_python.fg_if import FDMConnection

        fdm_conn = FDMConnection(fdm_version=24)  # adjust version if needed
        fdm_conn.connect_rx('localhost', 5501, self._fdm_callback)
        fdm_conn.connect_tx('localhost', 5502)
        fdm_conn.start()

        print("Streaming to FlightGear — press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.020)
        except KeyboardInterrupt:
            print("FlightGear stream stopped.")

    # --------------------------------------------------------------------------
    # Top-level entry point
    # --------------------------------------------------------------------------

    def run(self):
        """Full pipeline: setup → simulate → plot → visualise in FlightGear."""

        print("Setting up system...")
        self.setup_system()

        print("Designing controller...")
        self.setup_controller()

        print("Configuring time settings...")
        self.setup_time()

        print(f"Running simulation ({self.limit} steps)...")
        self.run_simulation()

        print_summary(self)
        show_metrics(self)

        print("Starting FlightGear visualisation...")
        self.start_flightgear()
        
                
        # print("Showing metrics...")       # ← ADD
        # print_summary(self)               # ← ADD (prints KPIs to terminal)
        # show_metrics(self)                # ← ADD (opens the dashboard plot)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    sim = LongitudinalControl()
    sim.run()