import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

R_earth = 6.371e6
M_earth = 5.9724e24
mass_unit = 1/62.46  # GIZMO mass unit
gamma = 1.3          # approximate gamma for silicates/iron

snap_start = 0
snap_end   = 300

massratio = 0.5           #planet mass/Earth mass
datadir = f"data{massratio*100:.0f}"
plotdir = f"plot{massratio*100:.0f}"
# -------- loop snapshots --------
for i in range(snap_start, snap_end + 1):
    # fname = 'n85_gizmo.hdf5'
    fname = f"{datadir}/snapshot_{i:03d}.hdf5"
    outname = f"{plotdir}/frame_{i:03d}.png"

    if not os.path.exists(fname):
        print(f"Missing {fname}, skip")
        continue
    if os.path.exists(outname):
        print(f"Existing {outname}, skip")
        continue
    with h5py.File(fname,'r') as f:
        pos = f['/PartType0/Coordinates'][()]   # code units
        m = f['/PartType0/Masses'][()]         # code units
        temp = f['/PartType0/Temperature'][()]
        compid = f['/PartType0/CompositionType'][()]

        entr = f['/PartType0/Entropy'][()]
        u = f['/PartType0/InternalEnergy'][()]
        time = f["Header"].attrs["Time"]
    # Convert to physical units
    r = np.linalg.norm(pos, axis=1)
    m_phys = m * mass_unit    

    # Sort by radius for cumulative mass
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    m_sorted = m_phys[sort_idx]

    m_enc = np.cumsum(m_sorted)
    unique_comp = np.unique(compid)

    # -------- plotting --------
    fig, ax = plt.subplots(2, 2, figsize=(10,8))
    fig.suptitle(f"Relaxing\nTime = {time:.4f}")

    # --- (1) enclosed mass ---
    for compi in unique_comp:
        mask = compid[sort_idx] == compi
        ax[0,0].plot(r_sorted[mask], m_enc[mask], '.', ms=1)
    ax[0,0].set_xlabel(r"Radius [$R_\oplus]$")
    ax[0,0].set_ylabel(r"Enclosed Mass $[M_\oplus]$")

    # --- (2) temperature ---
    for compi in unique_comp:
        mask = compid == compi
        ax[0,1].scatter(r[mask], temp[mask], s=1, alpha=0.6)

    ax[0,1].set_xlabel(r"Radius [$R_\oplus]$")
    ax[0,1].set_ylabel("Temperature [K]")

    # --- (3) internal energy ---
    for compi in unique_comp:
        mask = compid == compi
        ax[1,0].scatter(r[mask], u[mask], s=1, alpha=0.6)

    ax[1,0].set_xlabel(r"Radius [$R_\oplus]$")
    ax[1,0].set_ylabel("Internal Energy")

    # --- (4) entropy ---
    for compi in unique_comp:
        mask = compid == compi
        ax[1,1].scatter(r[mask], entr[mask], s=1, alpha=0.6)

    ax[1,1].set_xlabel(r"Radius [$R_\oplus]$")
    ax[1,1].set_ylabel("Entropy")

    # --- limits ---
    for a in ax.flat:
        a.set_xlim(0, 1.1)

    ax[0,0].set_ylim(0, 0.9)
    ax[0,1].set_ylim(500, 5000)

    # log scale helps a LOT
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()
    print(f"Saved {outname}")

# make video
import subprocess as sp
sp.run(["rm", "relaxing"+".mp4" ])
cmd = [
    "ffmpeg",
    "-r", "10",
    "-i", plotdir+"/frame_%03d.png",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "relaxing"+".mp4"
]

sp.run(cmd, check=True)