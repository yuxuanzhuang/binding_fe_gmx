import numpy as np

# convert free energy barrier to rate
def convert_barrier_2_rate(barrier_energy_kcal, temperature=300):
    """
    Convert a free energy barrier to a rate constant.
    Parameters
    ----------
    barrier_energy_kcal : float
        The free energy barrier in kcal/mol.
    temperature : float, optional, default=300
        The temperature in Kelvin.
    Returns
    -------
    rate : float
        The rate constant in 1/seconds.
    """
    R = 8.31446261815324 # J/(mol K)
    k_b = 1.38064852e-23 # J/K
    h = 6.62607004e-34 # J s

    # convert to kJ/mol
    barrier_energy_kj = barrier_energy_kcal * 4.184
    # convert to J/mol
    barrier_energy_j = barrier_energy_kj * 1000
    k_off = (1 / h) * np.exp(barrier_energy_j / (-R * temperature)) * k_b * temperature
    return k_off

# get absolute binding free energy
def get_binding_constant(r, pmf, binding_cutoff, temperature=300):
    """
    K_eq = pi * integral_in_binding_site(exp(-beta * w(r)) dr)
    dG = -RT * ln(K_eq / 1661)
    Parameters
    ----------
    r: np.ndarray
        The distance in Å.
    pmf : np.ndarray
        The PMF in kJ/mol.
    binding_cutoff : float
        The binding cutoff in Å.
    temperature : float, optional, default=300
        The temperature in Kelvin.
    """
    kT = 0.00831446261815324 * temperature
    beta = 1 / kT
    w_r = pmf - pmf[-10]
    # get the binding region
    binding_region = r < binding_cutoff
    ax.plot(r, w_r, c='black')
    ax.plot(r[-10], w_r[-10], 'o', c='red')
    r = r[binding_region][1:]
    w_r = w_r[binding_region][1:]
    ax.plot(r, w_r, c='red')
    k_eq = np.pi * np.trapz(np.exp(-beta * w_r), r)
    dG_bind = -kT * np.log(k_eq / 1661.)
    print(f'K_eq = {k_eq}')
    print(f'dG_bind = {dG_bind} kJ/mol')
    print(f'dG_bind = {dG_bind / 4.194} kcal/mol')

    return k_eq, dG_bind