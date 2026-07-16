
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# METRIC CALCULATION
# =============================================================================

def compute_metrics(sim):

    k = sim._k_final
    dT = sim.dT

    TIME = np.arange(k) * dT

    Xl = sim.system.Xlogged[:, :k]
    Ul = sim.system.Ulogged[:, :k]

    XO = sim.system.XO
    R = sim._R_bar

    h = Xl[0]
    theta = Xl[1]
    u = Xl[2]
    alpha = Xl[3]
    q = Xl[4]

    throttle = Ul[0]
    elevator = Ul[1]

    ref_h = XO[0] + R[0][:k]
    ref_u = XO[2] + R[1][:k]

    error_h = ref_h - h
    error_u = ref_u - u

    rms_h = np.sqrt(np.mean(error_h**2))
    rms_u = np.sqrt(np.mean(error_u**2))

    control_norm = np.sqrt(throttle**2 + elevator**2)

    return {
        "TIME": TIME,
        "h": h,
        "theta": theta,
        "u": u,
        "alpha": alpha,
        "q": q,
        "throttle": throttle,
        "elevator": elevator,
        "ref_h": ref_h,
        "ref_u": ref_u,
        "error_h": error_h,
        "error_u": error_u,
        "rms_h": rms_h,
        "rms_u": rms_u,
        "control_norm": control_norm,
        "elevator_limit": np.rad2deg(sim.controller.g_con[2])
    }


def show_metrics(sim):

    m = compute_metrics(sim)
    T = m["TIME"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "legend.fontsize": 9
    })

    LW = 2.3

    def style_axis(ax, loc="best"):
        ax.grid(True, alpha=0.3)
        ax.legend(loc=loc, framealpha=0.9)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("LONGITUDINAL PERFORMANCE PLOTS",
                 fontsize=28,
                 fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42,
                           wspace=0.32)

    # Altitude
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(T,m["ref_h"],"--",lw=LW,label="Reference")
    ax1.plot(T,m["h"],lw=LW,label="Actual")
    ax1.set_title("Altitude Tracking")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Altitude [m]")
    style_axis(ax1)

    # Velocity
    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(T,m["ref_u"],"--",lw=LW,label="Reference")
    ax2.plot(T,m["u"],lw=LW,label="Actual")
    ax2.set_title("Velocity Tracking")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity [ft/s]")
    style_axis(ax2)

    # Errors
    ax3 = fig.add_subplot(gs[0,2])
    ax3.plot(T,m["error_h"],lw=LW,label=f'Altitude RMS={m["rms_h"]:.2f}')
    ax3.plot(T,m["error_u"],lw=LW,label=f'Velocity RMS={m["rms_u"]:.2f}')
    ax3.axhline(0,color="k",ls=":")
    ax3.set_title("Tracking Errors")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Error")
    style_axis(ax3,"lower left")

    # Throttle
    ax4 = fig.add_subplot(gs[1,0])
    ax4.plot(T,m["throttle"],lw=LW,label="Throttle")
    ax4.axhline(1.0,ls="--",label="Upper Limit")
    ax4.axhline(0.0,ls="--",label="Lower Limit")
    ax4.set_ylim(-0.05,1.05)
    ax4.set_title("Throttle Command")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel(r"$\delta_t$")
    style_axis(ax4)

    # Elevator
    ax5 = fig.add_subplot(gs[1,1])
    ax5.plot(T,np.rad2deg(m["elevator"]),lw=LW,label="Elevator")
    ax5.axhline(m["elevator_limit"],ls="--",label="+ Limit")
    ax5.axhline(-m["elevator_limit"],ls="--",label="- Limit")
    ax5.set_title("Elevator Command")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Deflection [deg]")
    style_axis(ax5)

    # AoA & Pitch Rate
    ax6 = fig.add_subplot(gs[1,2])
    ax6b = ax6.twinx()
    l1 = ax6.plot(T,np.rad2deg(m["alpha"]),lw=LW,label="AoA")
    l2 = ax6b.plot(T,np.rad2deg(m["q"]),"--",lw=LW,label="Pitch Rate")
    ax6.set_title("AoA and Pitch Rate")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("AoA [deg]")
    ax6b.set_ylabel("Pitch Rate [deg/s]")
    ax6.grid(True,alpha=0.3)
    lines = l1 + l2
    ax6.legend(lines,[l.get_label() for l in lines],loc="best")

    plt.tight_layout(rect=[0.02,0.03,0.98,0.95])
    plt.show()


def print_summary(sim):

    m = compute_metrics(sim)

    print("="*60)
    print("MPC PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Simulation length     : {len(m['TIME'])*sim.dT:.2f} s")
    print(f"Altitude RMS error    : {m['rms_h']:.2f} m")
    print(f"Velocity RMS error    : {m['rms_u']:.2f}")
    print(f"Max airspeed          : {np.max(m['u']):.2f}")
    print(f"Min airspeed          : {np.min(m['u']):.2f}")
    print(f"Max elevator command  : {np.max(np.abs(np.rad2deg(m['elevator']))):.2f} deg")
    print("="*60)
