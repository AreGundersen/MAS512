import os
import pandas as pd
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(here, "..", "webots", "results")   # Problem 4\webots\results
out_folder = here                                    
runs = {"baseline": "tab:red", "low_slip": "tab:blue"}
V_MIN = 0                                 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for name, color in runs.items():
    df = pd.read_csv(os.path.join(data_folder, name + ".csv"))
    ok = df[df.v_x.abs() > V_MIN]
    ax1.plot(ok.t, ok.slip * 100, color=color, label=f"{name}")
    ax2.plot(df.x, df.y, color=color, label=name)

ax1.set_xlabel("time [s]")
ax1.set_ylabel("longitudinal slip [%]")
ax1.set_title("Longitudinal slip")
ax1.set_ylim(-75, 40)                         
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title("Path")
ax2.set_xlim(-0.5, 4.5)

ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_folder, "slip_plot.png"), dpi=150)
plt.show()