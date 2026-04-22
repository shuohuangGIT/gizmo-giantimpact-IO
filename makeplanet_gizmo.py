# coding: utf-8
import woma
import h5py
import numpy as np
import matplotlib.pyplot as plt

R_earth = 6.371e6   # m
M_earth = 5.9724e24 # kg

def plot_spherical_profiles(planet, savefile="oneplanet_new.png"):
    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    
    ax[0, 0].plot(planet.A1_r / R_earth, planet.A1_rho)
    ax[0, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 0].set_ylabel(r"Density $\rho$ [kg m$^{-3}$]")
    ax[0, 0].set_yscale("log")
    
    ax[1, 0].plot(planet.A1_r / R_earth, planet.A1_m_enc / M_earth)
    ax[1, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[1, 0].set_ylabel(r"Enclosed Mass $[M_\oplus]$")
    
    ax[0, 1].plot(planet.A1_r / R_earth, planet.A1_P)
    ax[0, 1].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 1].set_ylabel(r"Pressure [Pa]")
    ax[0, 1].set_yscale("log")
    
    ax[1, 1].plot(planet.A1_r / R_earth, planet.A1_T)
    ax[1, 1].set_xlabel(r"Radius $[R_\oplus]$")
    ax[1, 1].set_ylabel(r"Temperature [K]")
    
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

def save_planet_gizmo_ic(
    planet,
    fname,
    mass_unit=1/62.46,
    length_unit=R_earth,
    energy_unit = 1e6,
    particle_mass=2.65089e-5,
    Mat_ids=None,
    EOS_ids=None
):
    """
    Save a WOMA planet as GIZMO ICs in HDF5 format, fully layer-agnostic.

    Parameters
    ----------
    planet : woma.Planet
        The WOMA planet object.
    fname : str
        Output HDF5 filename.
    mass_unit : float
        GIZMO mass unit conversion factor.
    particle_mass : float
        Base particle mass in GIZMO units for first layer.
    Mat_ids : list of int, optional
        Material IDs for each layer.
    EOS_ids : list of int, optional
        EOS library IDs for each layer.
    """
    import os
    if os.path.exists(fname):
        os.remove(fname)

    n_layers = len(planet.A1_idx_layer)

    # Default IDs if not provided
    if Mat_ids is None:
        Mat_ids = list(range(400, 400+n_layers))
    if EOS_ids is None:
        EOS_ids = list(range(60, 60+n_layers))

    # Compute particle numbers per layer
    resn = np.zeros(n_layers, dtype=int)
    resn[0] = int(planet.M / M_earth / mass_unit / particle_mass)

    # resn[0] = int(mplanet/M_earth*62.46/2.65089e-5)  #test
    # print(int(mplanet/M_earth*62.46/2.65089e-5), int(planet.M / M_earth / mass_unit / particle_mass))
    
    if n_layers > 1:
        coredge = planet.A1_idx_layer[:-1]
        cmbratio = planet.A1_rho[coredge] / planet.A1_rho[coredge+1]
        for i in range(1, n_layers):
            resn[i] = int(resn[i-1] / cmbratio[i-1])

    # print(resn[1])          #test
    # resn[1] = 920000        #test

    # Generate particle distributions per layer
    resDiff = [woma.ParticlePlanet(planet, n, verbosity=0) for n in resn[::-1]]

    # Initialize arrays
    pos_list, vel_list, mass_list, u_list, temp_list = [], [], [], [], []
    ids_list, imat_list = [], []
    offset = 0

    # Loop over layers (the order is from the core to the mantle)
    for i in range(n_layers):
        mask = resDiff[i].A1_mat_id == Mat_ids[i]

        # Positions and velocities
        pos_list.append(resDiff[i].A2_pos[mask])
        vel_list.append(resDiff[i].A2_vel[mask])

        # Masses
        m_layer = resDiff[i].A1_m[mask].copy()
        m_layer[:] = planet.A1_M_layer[i] / M_earth / mass_unit / len(m_layer)

        mass_list.append(m_layer)

        # Internal energy and temperature
        u_list.append(resDiff[i].A1_u[mask])
        temp_list.append(resDiff[i].A1_T[mask])

        # Particle IDs
        ids_layer = np.arange(len(m_layer)) + offset
        ids_list.append(ids_layer)

        # EOS/material IDs
        imat_layer = np.full(len(m_layer), EOS_ids[i], dtype=int)
        imat_list.append(imat_layer)

        offset += 2*len(m_layer)

    # Concatenate all layers
    pos = np.concatenate(pos_list) / length_unit
    vel = np.concatenate(vel_list)
    m = np.concatenate(mass_list)
    u = np.concatenate(u_list) / energy_unit   # convert WOMA units -> GIZMO internal energy
    temp = np.concatenate(temp_list)
    ids = np.concatenate(ids_list)
    imat = np.concatenate(imat_list)
    entr = np.zeros(len(ids))           # entropy (zeros)

    # Write HDF5
    Ngas = len(ids)
    with h5py.File(fname, 'w') as f:
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
        p.create_dataset("CompositionType", data=imat)
        p.create_dataset("Masses", data=m)
        p.create_dataset("InternalEnergy", data=u)
        p.create_dataset("Temperature", data=temp)
        p.create_dataset("Entropy", data=entr)

        f.flush()

    print(f"Saved GIZMO IC to '{fname}', total mass = {np.sum(m)*mass_unit*M_earth/M_earth:.5f} M_earth")

if __name__ == "__main__":
    # Define planet properties
    mplanet = 0.5 * M_earth

    # Create WOMA planet with two layers: iron core + forsterite mantle
    planet = woma.Planet(
        name="Earth",
        A1_mat_layer=["ANEOS_iron", "ANEOS_forsterite"],
        A1_T_rho_type=["adiabatic", "adiabatic"],
        A1_M_layer=[0.3 * mplanet, 0.7 * mplanet],
        P_s=1e5,   # surface pressure [Pa]
        T_s=2000   # surface temperature [K]
    )

    # Generate radial profile within reasonable bounds
    planet.gen_prof_L2_find_R_R1_given_M1_M2(
        R_min=0.8 * R_earth, # (mplanet/M_earth)**0.25 *0.95
        R_max=0.9 * R_earth  # (mplanet/M_earth)**0.25 *1.08
    )

    # Optional: plot density, mass, pressure, temperature profiles
    plot_spherical_profiles(planet)

    # Save the planet as GIZMO initial conditions
    fname = f"n{mplanet/M_earth*100:.0f}.hdf5"
    save_planet_gizmo_ic(
        planet,
        fname,
        Mat_ids = [401, 400],  # inner to outer layer
        EOS_ids = [63, 62]  # inner to outer layer
    )

    print(f"Planet IC saved to '{fname}'")