# coding: utf-8
import numpy as np
import h5py
import woma

R_earth = 6.371e6
M_earth = 5.9724e24
G = 6.67408e-11

def load(fname):
    with h5py.File(fname, "r") as f:
        g = f["PartType0"]
        return {k: g[k][()] for k in g.keys()}

def center_of_mass(pos, m):
    return (pos * m[:, None]).sum(axis=0) / m.sum()

def combine_planets_gizmo_ic(fname1, fname2, fnameo, rsep, vimp,
                              mass_unit=(1/62.46)*M_earth,
                              length_unit=R_earth,
                              vel_unit=1e3):

    p1, p2 = load(fname1), load(fname2)

    def props(p):
        r = np.linalg.norm(p["Coordinates"], axis=1)
        M = p["Masses"].sum() * mass_unit
        R = r.max() * length_unit
        return M, R

    M1, R1 = props(p1)
    M2, R2 = props(p2)

    v_esc = np.sqrt(2 * G * (M1 + M2) / (R1 + R2))

    pos_i, vel_i = woma.impact_pos_vel_b_v_c_r(
        b=np.sin(np.deg2rad(45)),
        v_c= vimp*v_esc,
        r=rsep * (R1 + R2),
        R_t=R1, R_i=R2,
        M_t=M1, M_i=M2,
    )

    # move p2
    p2["Coordinates"] += pos_i / length_unit
    p2["Velocities"] += vel_i / vel_unit

    # temp combination
    pos_all = np.concatenate([p1["Coordinates"], p2["Coordinates"]])
    vel_all = np.concatenate([p1["Velocities"], p2["Velocities"]])
    m_all   = np.concatenate([p1["Masses"], p2["Masses"]])

    # COM correction (position)
    com_pos = (pos_all * m_all[:, None]).sum(axis=0) / m_all.sum()
    p1["Coordinates"] -= com_pos
    p2["Coordinates"] -= com_pos

    # Momentum correction (velocity)
    com_vel = (vel_all * m_all[:, None]).sum(axis=0) / m_all.sum()
    p1["Velocities"] -= com_vel
    p2["Velocities"] -= com_vel

    # merge
    def cat(key):
        return np.concatenate([p1[key], p2[key]])

    pos = cat("Coordinates")
    vel = cat("Velocities")
    ids = np.concatenate([p1["ParticleIDs"], p2["ParticleIDs"] + len(p1["ParticleIDs"])])
    comp = cat("CompositionType")
    m = cat("Masses")
    u = cat("InternalEnergy")
    temp = cat("Temperature")
    entr = cat("Entropy")

    # Write HDF5
    Ngas = len(ids)
    with h5py.File(fnameo, 'w') as f:
        npart = np.array([Ngas,0,0,0,0,0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart
        h.attrs['NumPart_Total'] = npart
        h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.int32)
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1.0
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['Omega0'] = 0.0
        h.attrs['OmegaLambda'] = 0.0
        h.attrs['HubbleParam'] = 0.0
        h.attrs['Flag_Sfr'] = 0
        h.attrs['Flag_Cooling'] = 0
        h.attrs['Flag_StellarAge'] = 0
        h.attrs['Flag_Metals'] = 0
        h.attrs['Flag_Feedback'] = 0
        h.attrs['Flag_DoublePrecision'] = 0
        h.attrs['Flag_IC_Info'] = 0

        p = f.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos)
        p.create_dataset("Velocities", data=vel)
        p.create_dataset("ParticleIDs", data=ids)
        p.create_dataset("CompositionType", data=comp)
        p.create_dataset("Masses", data=m)
        p.create_dataset("InternalEnergy", data=u)
        p.create_dataset("Temperature", data=temp)
        p.create_dataset("Entropy", data=entr)

        f.flush()

    print(f"Combine successfully! R_sep =",rsep)

if __name__ == "__main__":

    fname1 = "data50/snapshot_317.hdf5"
    fname2 = fname1
    fnameo = "n50ic.hdf5"

    combine_planets_gizmo_ic(
        fname1,
        fname2,
        fnameo,
        10,
        1
    )