import os
import glob
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Union, Dict, Any, Optional

from MDAnalysis.analysis.base import Results

from .utils.dhdl import AWH_DHDL
from .utils.xvg import XVG, AWH_1D_XVG, AWH_2D_XVG, AWH_3D_XVG
from .utils.log import AWH_LOG

class AWH_Ensemble(object):
    """
    AWH ensemble base class.
    It stores the awh results from `gmx awh` command as well as the pulling data.
    """
    def __init__(self,
                 folder: str,
                 replicate_prefix: str = 'rep',
                 pullx_file: str = 'awh_pullx.xvg',
                 pullf_file: str = 'awh_pullf.xvg',
                 dhdl_file: str = 'dhdl.xvg',
                 log_file: str = 'awh.log',
                 awh_result_folder: str = 'awh_result',
                 results_more=True,
                 stride: int = 1,
                 regenerate_awh: bool = False,
                 regenerate_dhdl: bool = False,
                 temperature: float = 300.0,
                 tmp=True,
                 ):
        """
        Parameters
        ----------
        folder : str
            The folder that contains the awh results.
            The structure of the folder should be:
            folder
            ├── rep1
            │   ├── awh_pullx.xvg
            │   ├── awh_pullf.xvg
            │   ├── dhdl.xvg
            │   └── awh.log
            ├── rep2
            │   ├── ...
        replicate_prefix : str, optional
            The prefix of the replicate folders, by default 'rep'
        pullx_file : str, optional
            The name of the pullx file, by default 'awh_pullx.xvg'
        pullf_file : str, optional
            The name of the pullf file, by default 'awh_pullf.xvg'
        dhdl_file : str, optional
            The name of the dhdl file, by default 'dhdl.xvg'
        log_file : str, optional
            The name of the log file, by default 'awh.log'
        awh_result_folder : str, optional
            The name of the folder that contains the awh results, by default 'awh_result'
        results_more : bool, optional
            Whether awh results contains more results
            (from `gmx awh -more`), by default True
        stride : int, optional
            The stride of the pulling data, by default 1
        regenerate_awh : bool, optional
            Whether to regenerate the awh results, by default False
        regenerate_dhdl : bool, optional
            Whether to regenerate the dhdl results, by default False
        temperature : float, optional
            The temperature of the simulation to set up kT value,
            by default 300.0
        tmp : bool, optional
            Whether to create a tmp folder no matter the folder can be written or not,
            by default False
        """
        self.folder = folder

        # if folder cannot be found, raise error
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f'{self.folder} not found.')

        # convert folder to absolute path
        self.folder = os.path.abspath(self.folder)
        self.tmp = tmp
        
        # if folder cannot be written, create a tmp folder
        if not os.access(self.folder, os.W_OK) or self.tmp:
            basename = os.path.basename(self.folder)
            # warning
            print(f'WARNING: {self.folder} cannot be written.')
            print(f'         or tmp=True is set.')
            print(f'         Creating a tmp folder {basename}_tmp.')
            os.makedirs(f'{basename}_tmp', exist_ok=True)
            self.write_folder = f'{basename}_tmp'
        else:
            self.write_folder = self.folder
            
        self.replicate_prefix = replicate_prefix
        self.awh_result_folder = awh_result_folder
        self.pullx_file = pullx_file
        self.pullf_file = pullf_file
        self.dhdl_file = dhdl_file
        self.log_file = log_file
        self.awh_prefix = self.log_file.split('.')[0]
        self.stride = stride
        self.results_more = results_more
        self.regenerate_awh = regenerate_awh
        self.regenerate_dhdl = regenerate_dhdl
        if self.regenerate_dhdl:
            raise ValueError('regenerate_dhdl is not implemented yet.')
        self.temperature = temperature
#        if self.regenerate_awh:
#            self._regenerate_awh()
#        if self.regenerae_dhdl:
#            self._generate_dhdl_files()

        self.awh_results = Results()
        self.awh_results.timeseries = []
        self.awh_results.pmf = []
        self.awh_pmf = []

        self._gather_data()

        self._generate_pulling_data()

        if not self.no_dhdl_file:
            self._generate_dhdl_data()
        self._generate_log_data()
        self._generate_pmf_data()
    
    def _gather_data(self):
        self.rep_folder = []
        self.awh_pullx_files = []
        self.awh_pullf_files = []
        self.awh_log_files =[]
        self.awh_dhdl_files = []
        for folder in glob.glob(f'{self.folder}/{self.replicate_prefix}*'):
            self.rep_folder.append(folder)
            self.awh_pullx_files.append(folder + '/' + self.pullx_file)
            self.awh_pullf_files.append(folder + '/' + self.pullf_file)
            self.awh_log_files.append(folder + '/' + self.log_file)
            self.awh_dhdl_files.append(folder + '/' + self.dhdl_file)

        if len(self.rep_folder) == 0:
            raise ValueError(f'No replicate folders found in {self.folder}.')
            
        self.rep_folder.sort(key=lambda x: eval(x.split('/')[-1][-1])) # sort by replicate number
        self.awh_pullx_files.sort(key=lambda x: eval(x.split('/')[-2][-1])) # sort by replicate number
        self.awh_pullf_files.sort(key=lambda x: eval(x.split('/')[-2][-1])) # sort by replicate number
        self.awh_log_files.sort(key=lambda x: eval(x.split('/')[-2][-1])) # sort by replicate number
        self.awh_dhdl_files.sort(key=lambda x: eval(x.split('/')[-2][-1])) # sort by replicate number

        self.awh_pmf_files = []
        for f in glob.glob(f'{self.folder}/{self.awh_result_folder}/{self.awh_prefix}*'):
            self.awh_pmf_files.append(f)
        
        if len(self.awh_pmf_files) == 0:
            self._regenerate_awh()

        # sort by time value
        self.awh_pmf_files.sort(key=lambda x: eval(x.split('/')[-1].split('.')[0].split('_')[1][1:]))

        # check for empty list
        if len(self.rep_folder) == 0:
            raise ValueError('No rep folder found. Check the folder name.')

        # make sure every file exists
        for f in self.awh_pmf_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f} not found.')
        for f in self.awh_pullx_files:
            if not os.path.exists(f):
                print(f'WARNING: {f} not found.')
        for f in self.awh_pullf_files:
            if not os.path.exists(f):
                print(f'WARNING: {f} not found.')
        for f in self.awh_log_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f} not found.')
        for f in self.awh_dhdl_files:
            if not os.path.exists(f):
#                print('WARNING: No awh_dhdl file found. Check the folder name.')
                self.no_dhdl_file = True
                break
            else:
                self.no_dhdl_file = False

        if self.no_dhdl_file and self.regenerate_dhdl:
            self._generate_dhdl_files()
            self.no_dhdl_file = False

        last_awh_pmf_files_time = os.path.getmtime(self.awh_pmf_files[-1])
        last_rep_folder_time = os.path.getmtime(self.awh_pullx_files[-1])
        if last_rep_folder_time > last_awh_pmf_files_time:
            if self.regenerate_awh:
                self._regenerate_awh()

            else:
                print('WARNING: The latest walker was generated after the latest awh_pmf file.')
                print('         add `regenerate_awh=True`.')

        print(f'Found {len(self.awh_pmf_files)} awh_pmf files.')
        print(f'The latest awh_pmf file is {self.awh_pmf_files[-1]}')
        last_awh_pmf_files_time = datetime.fromtimestamp(last_awh_pmf_files_time)
        print(f'The latest awh_pmf file was generated at {last_awh_pmf_files_time}')
        print(f'Found {len(self.rep_folder)} walkers.')
        last_rep_folder_time = datetime.fromtimestamp(last_rep_folder_time)
        print(f'The latest walker was generated at {last_rep_folder_time}')


    def _generate_pulling_data(self):
        print('Generating pulling data...')
        self.awh_pullx = []
        self.awh_pullf = []
        for walker_index, awh_walker in enumerate(self.awh_pullx_files):
            self.awh_pullx.append(XVG(awh_walker, index=walker_index+1))
        for walker_index, awh_walker in enumerate(self.awh_pullf_files):
            self.awh_pullf.append(XVG(awh_walker, index=walker_index+1))

    def _generate_pmf_data(self):
        print('Generating PMF data...')
        for awh_files in tqdm(self.awh_pmf_files[::self.stride],
                              desc='Generating PMF data',
                              total=len(self.awh_pmf_files[::self.stride])):
            time, unit, awh_pmf, awh_pmf_xvg = self.get_awh_pmf(awh_files, self.results_more)
            self.awh_results.timeseries.append(time)
            self.awh_results.pmf.append(awh_pmf)
            self.awh_pmf.append(awh_pmf_xvg)
            self.unit = unit

            self.sample_weights = []
            self.est_error = []
            for pmf in self.awh_pmf:
                self.sample_weights.append(pmf.log_sample_weight)
                self.est_error.append(pmf.taget_error)

            self.timeseries = []
            for time in self.awh_results.timeseries:
                self.timeseries.append(eval(time[1:]))

    def _generate_dhdl_data(self):
        print('Generating dH/dl data...')
        self.awh_dhdl = []
        for awh_dhdl_file in self.awh_dhdl_files:
            self.awh_dhdl.append(AWH_DHDL(awh_dhdl_file))

    def _generate_log_data(self):
        print('Generating log data...')
        self.awh_log = []
        for awh_log_file in self.awh_log_files:
            self.awh_log.append(AWH_LOG(awh_log_file))

    def _regenerate_awh(self):
        import gromacs
        cwd = os.getcwd()
        os.makedirs(f'{self.write_folder}/{self.awh_result_folder}', exist_ok=True)
        os.chdir(f'{self.write_folder}/{self.awh_result_folder}')
        # remove xvg files from previous run
        for file in glob.glob('*.xvg'):
            os.remove(file)
        awh_command = gromacs.awh(f=f'{self.folder}/{self.replicate_prefix}1/{self.awh_prefix}.edr',
                        o=f'{self.awh_prefix}.xvg',
                        s=f'{self.folder}/{self.replicate_prefix}1/{self.awh_prefix}.tpr',
                        skip=10,
                        more=True)
        # remove files starting with #
        for file in glob.glob('#*'):
            os.remove(file)
        os.chdir(cwd)
        self.awh_pmf_files = []
        for f in glob.glob(f'{self.write_folder}/{self.awh_result_folder}/{self.awh_prefix}*'):
            self.awh_pmf_files.append(f)
        self.awh_pmf_files.sort(key=lambda x: eval(x.split('/')[-1].split('.')[0].split('_')[1][1:]))


    def _generate_dhdl_files(self):
        raise NotImplementedError('dhdl cannot be generated yet.')
        import gromacs
        cwd = os.getcwd()

        for dhdl_file in self.awh_dhdl_files:
            dhdl_file_dir = os.path.dirname(dhdl_file)
            os.chdir(dhdl_file_dir)
            awh_command = gromacs.energy(f=f'{self.awh_prefix}.edr',
                            odh=f'{os.path.basename(dhdl_file)}',
                            s=f'{self.awh_prefix}.tpr')
            # remove files starting with #
            for file in glob.glob('#*'):
                os.remove(file)
            os.chdir(cwd)
            
    def generate_pmf_video(self,
                           name,
                           stride=1,
                           levels=None,
                           vmax=None,
                           cmap='coolwarm',
                           pmf_label='PMF',
                           marginalize_cv=None,
                           remove_img=True,
                           ffmpeg='ffmpeg',
                           **kwargs):
        """
        Generate a video of the PMF time evolution.

        Parameters
        ----------
        name : str
            The name of the video.
            saved under the folder `video/`.
        stride : int
            The stride of the time series.
        levels: np.array
            The levels to plot the PMF.
            Default is None
        vmax: float
            The max value to plot.
            Default is None
        cmap: str
            colormap
            Default is 'coolwarm'
        pmf_label: str
            The label of the PMF.
            Default is 'PMF'
        marginalize_cv: int
            The CV that will be marginalized.
        remove_img : bool
            Whether to remove the images after the video is generated.
            Default is True.
        ffmpeg : str
            The path to the ffmpeg executable.
        """

        os.makedirs(self.write_folder + '/video/', exist_ok=True)
        from tqdm import tqdm
        for i, awh_pmf in tqdm(enumerate(self.awh_results.pmf[::stride]),
                            total=len(self.awh_results.pmf[::stride])):
            fig, ax = plt.subplots(figsize=(7,9))
            time = self.awh_results.timeseries[i*stride]

            _, mappable, plot_cbar = self.plot_pmf(awh_pmf, timeseries=time, ax=ax,
                                                   levels=levels, vmax=vmax, cmap=cmap, kT=self.kT,
                                                   pmf_label=pmf_label,
                                                   marginalize_cv=marginalize_cv,
                                                   **kwargs)
            if plot_cbar:
                cbar = fig.colorbar(mappable)
                cbar.set_label(f'{pmf_label}')

            time_ns = eval(time[1:]) / 1000
            ax.set_title(f'{time_ns:.0f} ns {pmf_label}', fontsize=20)
            plt.savefig(f'{self.write_folder}/video/{name}_{i}.png')
            plt.close()
        
        # generate video with ffmpeg

        ffmpeg_command = [ffmpeg,
                '-y',
                '-framerate', '5',
                '-i', f'{name}_%d.png',
                '-r', '30',
                '-b', '5000k',
                '-vcodec', 'mpeg4',
                f'{name}.mp4']

        process = subprocess.Popen(ffmpeg_command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=0,
                        cwd=self.write_folder + '/video/')
        process.stdin.close()

        for line in process.stdout:
            print(line.strip())
        for line in process.stderr:
            print(line.strip())
        
        if remove_img:
            os.system(f'rm {self.write_folder}/video/{name}_*.png')
    
    def __repr__(self) -> str:
        return f'AWH_Ensemble(replicate_prefix={self.replicate_prefix}, stride={self.stride}, results_more={self.results_more})'

    @property
    def kT(self):
        """
        Energy conversion factor.
        """
        if self.unit == 'kJ/mol':
            return self.temperature * 0.00831446261815324
        elif self.unit == 'kcal/mol':
            return self.temperature * 0.0019872041
        elif self.unit == 'kT':
            return 1
        elif self.unit == 'unknown':
            return 1
        else:
            raise ValueError('Unknown unit.')

    @staticmethod
    def get_awh_pmf(awh_file, results_more):
        raise NotImplementedError('Use AWH_1D_Ensemble, AWH_2D_Ensemble, or AWH_3D_Ensemble instead.')

    @staticmethod
    def plot_pmf(awh_pmf, timeseries=None, ax=None, **kwargs):
        raise NotImplementedError('Use AWH_1D_Ensemble, AWH_2D_Ensemble, or AWH_3D_Ensemble instead.')   


class AWH_1D_Ensemble(AWH_Ensemble):
    """
    AWH ensemble class for 1D PMF.
    """
    @staticmethod
    def get_awh_pmf(awh_file, results_more):
        """
        Returns the 1D PMF of the AWH file.
        awh_file: str
            Path to the awh file.
        results_more: bool
            Whether the awh file contains more information than just the PMF.

        Returns
        -------
        time: str
            The time of the awh file.
        unit: str
            The unit of the PMF.
        awh_pmf: np.array
            The PMF.
        """
        time = awh_file.split('/')[-1].split('.')[0].split('_')[-1]

        if results_more:
            awh_pmf_xvg = AWH_1D_XVG(awh_file, custom_names=[
                        'dim1',
                        'PMF', 'Coord_bias', 'Coord_distr',
                        'Ref_value_distr', 'Target_ref_value_distr',
                        'Friction_metric'])
        else:
            awh_pmf_xvg = AWH_1D_XVG(awh_file, custom_names=[
                        'dim1',
                        'PMF'])        
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf, awh_pmf_xvg

    @staticmethod
    def plot_pmf(awh_pmf, timeseries=None, ax=None, **kwargs):
        """
        Plot the 1D PMF.

        Parameters
        ----------
        awh_pmf: np.array
            The PMF generated from `gmx awh`.
        timeseries : np.array, optional
            The time series of the PMF, by default None
        ax : plt.axes, optional
            The axes to plot the PMF, by default None
        **kwargs : dict
            Other arguments to pass to plt.plot
        """

        cv_labels = kwargs.pop('cv_labels', ['CV1'])
        pmf_label = kwargs.pop('pmf_label', 'PMF')
        awh_cv1 = awh_pmf.T[0]
        awh_fes = awh_pmf[:,2].T

        if ax is None:
            fig, ax = plt.subplots()
        if timeseries is None:
            mappable = ax.plot(awh_cv1, awh_fes, **kwargs)
        else:
            mappable = ax.plot(awh_cv1, awh_fes, label=f'{timeseries}', **kwargs)
            ax.legend()
        ax.set_xlabel(cv_labels[0])

        ax.set_ylabel(pmf_label)
        return ax, mappable, False
    
class AWH_2D_Ensemble(AWH_Ensemble):
    """
    AWH ensemble class for 2D PMF.
    """
    @staticmethod
    def get_awh_pmf(awh_file, results_more):
        """
        Returns the 2D PMF of the AWH file.
        """
        time = awh_file.split('/')[-1].split('.')[0].split('_')[-1]

        if results_more:
            awh_pmf_xvg = AWH_2D_XVG(awh_file, custom_names=[
                        'dim1', 'dim2',
                        'PMF', 'Coord_bias', 'Coord_distr',
                        'Ref_value_distr', 'Target_ref_value_distr',
                        'Friction_metric'])
        else:
            awh_pmf_xvg = AWH_2D_XVG(awh_file, custom_names=[
                        'dim1', 'dim2',
                        'PMF'])
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf, awh_pmf_xvg

    @staticmethod
    def plot_pmf(awh_pmf, timeseries=None, ax=None, **kwargs):
        """
        Plot the 2D PMF.

        Parameters
        ----------
        awh_pmf: np.array
            The PMF generated from `gmx awh`.
        timeseries : np.array, optional
            The time series of the PMF, by default None
        ax : plt.axes, optional
            The axes to plot the PMF, by default None
        **kwargs : dict
            Other arguments to pass to plt.plot
        """

        cmap = kwargs.pop('cmap', 'coolwarm')
        vmax = kwargs.pop('vmax', None)
        levels = kwargs.pop('levels', None)
        cv_labels = kwargs.pop('cv_labels', ['CV1', 'CV2'])

        awh_cv1 = awh_pmf.T[0][0]
        awh_cv2 = awh_pmf[0].T[1]
        awh_fes = awh_pmf[:,:,2].T
        if ax is None:
            fig, ax = plt.subplots()

        mappable = ax.contourf(
            awh_cv1,
            awh_cv2,
            awh_fes,
            cmap=cmap,
            vmax=vmax,
            levels=levels,
            extend='both')
        
        ax.set_xlabel(cv_labels[0])
        ax.set_ylabel(cv_labels[1])

        return ax, mappable, True
    

class AWH_3D_Ensemble(AWH_Ensemble):
    """
    AWH ensemble class for 3D PMF.
    """
    @staticmethod
    def get_awh_pmf(awh_file, results_more):
        """
        Returns the 3D PMF of the AWH file.
        """
        time = awh_file.split('/')[-1].split('.')[0].split('_')[-1]

        if results_more:
            awh_pmf_xvg = AWH_3D_XVG(awh_file, custom_names=[
                        'dim1', 'dim2', 'dim3',
                        'PMF', 'Coord_bias', 'Coord_distr',
                        'Ref_value_distr', 'Target_ref_value_distr',
                        'Friction_metric'])
        else:
            awh_pmf_xvg = AWH_3D_XVG(awh_file, custom_names=[
                        'dim1', 'dim2', 'dim3',
                        'PMF'])
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf, awh_pmf_xvg
    

    @staticmethod
    def plot_pmf(awh_pmf, timeseries=None, ax=None, **kwargs):
            
        cmap = kwargs.pop('cmap', 'coolwarm')
        vmax = kwargs.pop('vmax', None)
        levels = kwargs.pop('levels', None)
        kT = kwargs.pop('kT', 1)
        # integral over CV3
        marginalize_cv = kwargs.pop('marginalize_cv', 2)
        if marginalize_cv not in [0, 1, 2]:
            raise ValueError('marginalize_cv should be 0, 1, or 2.')
        
        cv_labels = kwargs.pop('cv_labels', ['CV1', 'CV2', 'CV3'])
        rm_cv_lab = cv_labels.pop(marginalize_cv)

#        print(f'Marginalizing {rm_cv_lab}...')
    
        awh_cv1 = awh_pmf.T[0][0][0]
        awh_cv2 = awh_pmf.transpose(0,3,2,1)[0][1][0]
        awh_cv3 = awh_pmf[0].T[2].T[0]
        awh_fes = awh_pmf[:,:,:,3].T

        all_cvs = [awh_cv1, awh_cv2, awh_cv3]
        int_cv = all_cvs.pop(marginalize_cv)

        awh_fes_int = np.zeros((all_cvs[1].shape[0], all_cvs[0].shape[0]))
#        awh_fes_marg = np.moveaxis(awh_fes, marginalize_cv, 2)
        
        # for now just hack the code
        if marginalize_cv == 0:
            awh_fes_marg  = awh_fes.transpose(2,0,1)
        elif marginalize_cv == 1:
            awh_fes_marg = awh_fes.transpose(1,0,2)
        elif marginalize_cv == 2:
            awh_fes_marg = awh_fes
        else:
            raise ValueError('marginalize_cv should be 0, 1, or 2.')

        for i in range(all_cvs[1].shape[0]):
            for j in range(all_cvs[0].shape[0]):
                awh_fes_int[i,j] = np.trapz(np.exp(-awh_fes_marg[:, i, j] / kT), int_cv)
        awh_fes_int = -np.log(awh_fes_int) * kT

        if ax is None:
            fig, ax = plt.subplots()
        
        mappable = ax.contourf(
                    all_cvs[0],
                    all_cvs[1],
                    awh_fes_int,
                    cmap=cmap,
                    vmax=vmax,
                    levels=levels,
                    extend='both')
        
        ax.set_xlabel(cv_labels[0])
        ax.set_ylabel(cv_labels[1])

        return ax, mappable, True