import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from ENPMDA import MDDataFrame
from ENPMDA.preprocessing import TrajectoryEnsemble
from ENPMDA.analysis.base import DaskChunkMdanalysis
from MDAnalysis.analysis.distances import distance_array

next_subunit_dict = {'A': 'C',
                     'B': 'D',
                     'C': 'E',
                     'D': 'F',
                     'E': 'G',
                     'F': 'H',
                     'G': 'I',
                     'H': 'J',
                     'I': 'A',
                     'J': 'B'}

prot_selections = ['protein and ((resid 100 and segid A) or (resid 155 and segid A) or (resid 156 and segid A) or (resid 157 and segid A) or (resid 197 and segid A) or (resid 199 and segid A) or (resid 200 and segid A) or (resid 204 and segid A) or (resid 57 and segid C) or (resid 81 and segid C) or (resid 109 and segid C) or (resid 110 and segid C) or (resid 111 and segid C) or (resid 119 and segid C) or (resid 121 and segid C))',
                   'protein and ((resid 100 and segid C) or (resid 155 and segid C) or (resid 156 and segid C) or (resid 157 and segid C) or (resid 197 and segid C) or (resid 199 and segid C) or (resid 200 and segid C) or (resid 204 and segid C) or (resid 57 and segid E) or (resid 81 and segid E) or (resid 109 and segid E) or (resid 110 and segid E) or (resid 111 and segid E) or (resid 119 and segid E) or (resid 121 and segid E))',
                   'protein and ((resid 100 and segid E) or (resid 155 and segid E) or (resid 156 and segid E) or (resid 157 and segid E) or (resid 197 and segid E) or (resid 199 and segid E) or (resid 200 and segid E) or (resid 204 and segid E) or (resid 57 and segid G) or (resid 81 and segid G) or (resid 109 and segid G) or (resid 110 and segid G) or (resid 111 and segid G) or (resid 119 and segid G) or (resid 121 and segid G))',
                   'protein and ((resid 100 and segid G) or (resid 155 and segid G) or (resid 156 and segid G) or (resid 157 and segid G) or (resid 197 and segid G) or (resid 199 and segid G) or (resid 200 and segid G) or (resid 204 and segid G) or (resid 57 and segid I) or (resid 81 and segid I) or (resid 109 and segid I) or (resid 110 and segid I) or (resid 111 and segid I) or (resid 119 and segid I) or (resid 121 and segid I))',
                   'protein and ((resid 100 and segid I) or (resid 155 and segid I) or (resid 156 and segid I) or (resid 157 and segid I) or (resid 197 and segid I) or (resid 199 and segid I) or (resid 200 and segid I) or (resid 204 and segid I) or (resid 57 and segid A) or (resid 81 and segid A) or (resid 109 and segid A) or (resid 110 and segid A) or (resid 111 and segid A) or (resid 119 and segid A) or (resid 121 and segid A))']

prot_resids = [100, 155, 156, 157, 197, 199, 200, 204, 57, 81, 109, 110, 111, 119, 121]
chain_resids = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C']

class get_loopc_dynamics(DaskChunkMdanalysis):
    name = 'loopC'
    universe_file = 'protein'

    def set_feature_info(self, universe):
        return ['chn_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        binding_sites = []
        ch_1_loopc = universe.select_atoms('segid A and resid 193-206 and name CA')
        ch_2_loopc = universe.select_atoms('segid C and resid 185-198 and name CA')
        ch_3_loopc = universe.select_atoms('segid E and resid 185-198 and name CA')
        ch_4_loopc = universe.select_atoms('segid G and resid 193-206 and name CA')
        ch_5_loopc = universe.select_atoms('segid I and resid 185-198 and name CA')

        for selection in prot_selections:
            binding_sites.append(universe.select_atoms(selection))

        loopC_sites = [ch_1_loopc, ch_2_loopc, ch_3_loopc, ch_4_loopc, ch_5_loopc]
        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(np.asarray([distance_array(loopc.center_of_mass(),
                                                     bs.center_of_mass())[0]
                                            for loopc, bs in zip(loopC_sites, binding_sites)]))
        return result
    

class get_epj_contact(DaskChunkMdanalysis):
    name = 'epj_contact'
    universe_file = 'protein'

    def set_feature_info(self, universe):
        return ['_'.join([str(res), chn]) for res, chn in zip(prot_resids, chain_resids)]

    def run_analysis(self, universe, start, stop, step):
        epj = universe.select_atoms('resname EPJ').split('residue')[0]
        binding_site = universe.select_atoms(prot_selections[0])

        result = []
        for ts in universe.trajectory[start:stop:step]:
            contacts = []
            for residue in binding_site.residues:
                contacts.append(np.min(distance_array(residue.atoms.positions, epj.positions)))
            result.append(contacts)
        return result
    

class get_water_dynamics(DaskChunkMdanalysis):
    name = 'water'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['num', 'bridge_108']

    def run_analysis(self, universe, start, stop, step):
        epj = universe.select_atoms('resname EPJ').split('residue')[0]
        water_around_epj = universe.select_atoms('resname TIP3 and around 4 group epj', epj=epj,
                                                 updating=True)
        
        res_108 = universe.select_atoms('resid 108 and segid C')
        water_around_108 = universe.select_atoms('resname TIP3 and around 4 group res_108', res_108=res_108,
                                                updating=True)
        
        
        result = []
        for ts in universe.trajectory[start:stop:step]:
            intersection = water_around_epj.intersection(water_around_108)
            result.append(np.asarray([len(water_around_epj.residues), len(intersection.residues)]))
        return result
    

def get_sample_weight(feature_dataframe, awh_ensemble, pulling_awh_cvs):
    """
    Get sample weights for each frame in the feature_dataframe.

    Parameters
    ----------
    feature_dataframe : pandas.DataFrame
        Dataframe containing the features for each frame.
    awh_ensemble : AWH_Ensemble
        AWH_Ensemble object containing the AWH results.
    pulling_awh_cvs : list
        List of names of the pulling cvs used in the AWH simulation.

    Returns
    -------
    samp_weights : numpy.ndarray
        Array containing the sample weights for each frame.
    """

    sample_weights = awh_ensemble.sample_weights
    timeseries = awh_ensemble.timeseries
    awh_pmf = awh_ensemble.awh_results.pmf[-1]
    awh_fes = awh_pmf.reshape(-1, 8).T[2]
    awh_cvs = awh_pmf.reshape(-1, 8).T[:2]

    df_timeseries = feature_dataframe.traj_time.to_numpy()

    # get closest index of timeseries for each df_timeseries based on difference
    g = np.argmin(np.abs(df_timeseries[:, np.newaxis] - timeseries), axis=1)
    timeweights = np.asarray(sample_weights)[g]

    pullx_traj = feature_dataframe[pulling_awh_cvs].to_numpy()
    # get the closest index of awh_cvs for each pullx_traj based on squared difference
    g = np.argmin(np.sum((awh_cvs.T[:, np.newaxis, :] - pullx_traj[np.newaxis, :, :])**2, axis=2), axis=0)     
    energyweights = awh_fes[g]

    normalization_weights = 1.0 - squareform(pdist(pullx_traj, "hamming"))
    normalization_weights = 1 / np.sum(normalization_weights,-1)

    """Retrieve total weights for each frame"""
    samp_weights = normalization_weights * np.exp(-energyweights / awh_ensemble.kT) * timeweights

    return samp_weights


# get absolute binding free energy

def get_binding_kinectics(r,
                          pmf,
                          binding_range,
                          ax,
                          r_star_ind=-10,
                          temperature=300,
                          printing=False):
    """
    K_eq = pi * integral_in_binding_site(exp(-beta * w(r)) dr)
    dG = -RT * ln(K_eq / 1661)
    Parameters
    ----------
    r: np.ndarray
        The distance in Å.
    pmf : np.ndarray
        The PMF in kJ/mol.
    binding_range : tuple
        The binding region in Å.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    r_star_ind : int, optional, default=-10
        The index of the r*.
        r* is the unbound area.
        W(r*) = −kT ln(α r * 2) must hold,
        α is a constant.
    temperature : float, optional, default=300
        The temperature in Kelvin.
    printing : bool, optional, default=False
        If True, print the results.
    """
    kT = 0.00831446261815324 * temperature
    beta = 1 / kT
    r_start = r[r_star_ind]
    w_r = pmf - pmf[r_star_ind]
    # get the binding region
    binding_region = np.where((r > binding_range[0]) & (r < binding_range[1]))

    ax.plot(r, w_r, c='black')
    ax.plot(r[r_star_ind], w_r[r_star_ind], 'o', c='red')

    r_binding = r[binding_region][1:]
    w_r_binding = w_r[binding_region][1:]
    ax.plot(r_binding, w_r_binding, c='red')
    k_eq = 2 * np.pi * r_start * 20 / 5
    k_eq = k_eq * np.trapz(np.exp(-beta * w_r_binding), r_binding)
    dG_bind = -kT * np.log(k_eq / 1661.)
    if printing:
        print(f'r* = {r_start} Å')
        print(f'K_eq = {k_eq}')
        print(f'dG_bind = {dG_bind} kJ/mol')
        print(f'dG_bind = {dG_bind / 4.194} kcal/mol')

    return k_eq, dG_bind