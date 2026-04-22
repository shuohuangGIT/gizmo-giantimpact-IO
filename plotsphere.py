import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import os

# Define start and end colors
start_color = mcolors.to_rgba('navy')  # 'tab:blue'
end_color = mcolors.to_rgba('firebrick')    # brown

# Create a custom colormap
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('BlueToBrown', [start_color, end_color])

snap_start = 0
snap_end   = 160

massratio = 0.5           #planet mass/Earth mass
datadir = f"data{massratio*100:.0f}ic"
plotdir = f"plot{massratio*100:.0f}ic"
# -------- loop snapshots --------
for i in range(snap_start, snap_end + 1):
    # fname = 'n85_gizmo.hdf5'
    fname = f"{datadir}/snapshot_{i:03d}.hdf5"
    outname = f"{plotdir}/sphere_{i:03d}.png"

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
        time = f["Header"].attrs["Time"]
            
    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)           # polar angle: 0 at north pole
    phi = np.arctan2(y, x)             # azimuthal angle: -pi to pi
    phi = np.mod(phi, 2*np.pi)         # convert to 0 - 2pi

    # Mask out the octant: theta in [0, pi/2], phi in [0, pi/2]
    octant_mask = ~((theta >= 0) & (theta <= np.pi/2) & (phi >= 0) & (phi <= np.pi/2))

    x_mask = x[octant_mask]
    y_mask = y[octant_mask]
    z_mask = z[octant_mask]
    compid_mask = compid[octant_mask]
    temp_mask = temp[octant_mask]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Camera view parameters
    elev, azim = 30, 30
    ax.view_init(elev=elev, azim=azim)

    # Scatter per component, sorted by distance along camera direction
    alphas = np.zeros_like(x_mask)

    unique_comp = np.unique(compid)
    for i, compi in enumerate(unique_comp):
        comp_indx = compid_mask == compi
        alphas[comp_indx] = 0.6-i*0.3

    norm = Normalize(vmin = 500, vmax = 5000, clip = False)
    ax.scatter(x_mask, y_mask, z_mask, c=temp_mask, alpha=alphas, s=0.05, cmap=custom_cmap)

    # Equal aspect ratio
    max_range = np.array([x_mask.max()-x_mask.min(),
                        y_mask.max()-y_mask.min(),
                        z_mask.max()-z_mask.min()])
    max_range_val = max(max_range)/2
    mid_x = (x_mask.max()+x_mask.min())*0.5
    mid_y = (y_mask.max()+y_mask.min())*0.5
    mid_z = (z_mask.max()+z_mask.min())*0.5
    ax.set_xlim(mid_x - max_range_val, mid_x + max_range_val)
    ax.set_ylim(mid_y - max_range_val, mid_y + max_range_val)
    ax.set_zlim(mid_z - max_range_val, mid_z + max_range_val)

    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')   # optional but recommended
    # ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()
    print(f"Saved {outname}")

#make video
import subprocess as sp
sp.run(["rm", "sphere_relaxing"+".mp4" ])
cmd = [
    "ffmpeg",
    "-r", "10",
    "-i", plotdir+"/sphere_%03d.png",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "sphere_relaxing"+".mp4"
]

sp.run(cmd, check=True)