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
    1D AWH ensemble class.
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
        """
        self.folder = folder
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
                raise FileNotFoundError(f'{f} not found.')
        for f in self.awh_pullf_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f} not found.')
        for f in self.awh_log_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f} not found.')
        for f in self.awh_dhdl_files:
            if not os.path.exists(f):
                print('WARNING: No awh_dhdl file found. Check the folder name.')
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
            time, unit, awh_pmf = self.get_awh_pmf(awh_files, self.results_more)
            self.awh_results.timeseries.append(time)
            self.awh_results.pmf.append(awh_pmf)
            self.unit = unit

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
        os.makedirs(f'{self.folder}/{self.awh_result_folder}', exist_ok=True)
        os.chdir(f'{self.folder}/{self.awh_result_folder}')
        # remove xvg files from previous run
        for file in glob.glob('*.xvg'):
            os.remove(file)
        awh_command = gromacs.awh(f=f'../{self.replicate_prefix}1/{self.awh_prefix}.edr',
                        o=f'{self.awh_prefix}.xvg',
                        s=f'../{self.replicate_prefix}1/{self.awh_prefix}.tpr',
                        skip=10,
                        more=True)
        # remove files starting with #
        for file in glob.glob('#*'):
            os.remove(file)
        os.chdir(cwd)
        self.awh_pmf_files = []
        for f in glob.glob(f'{self.folder}/{self.awh_result_folder}/{self.awh_prefix}*'):
            self.awh_pmf_files.append(f)
        self.awh_pmf_files.sort(key=lambda x: eval(x.split('/')[-1].split('.')[0].split('_')[1][1:]))


    def _generate_dhdl_files(self):
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
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf

    def generate_pmf_video(self,
                           name,
                           stride=1,
                           remove_img=True,
                           ffmpeg='ffmpeg'):
        """
        Generate a video of the PMF time evolution.

        Parameters
        ----------
        name : str
            The name of the video.
            saved under the folder `video/`.
        stride : int
            The stride of the time series.
        remove_img : bool
            Whether to remove the images after the video is generated.
            Default is True.
        ffmpeg : str
            The path to the ffmpeg executable.
        """

        os.makedirs(self.folder + '/video/', exist_ok=True)
        from tqdm import tqdm
        for i, awh_pmf in tqdm(enumerate(self.awh_results.pmf[::stride]),
                            total=len(self.awh_results.pmf[::stride])):
            time = self.awh_results.timeseries[i*stride]

            awh_cv1 = awh_pmf.T[0][0]
            awh_fes = awh_pmf[:,:,1].T

            fig, ax = plt.subplots(figsize=(7,9))
            mappable = ax.plot(
                        awh_cv1,
                        awh_fes,
        #                vmax=100,
                        levels=1000)

            ax.set_ylabel(f'PMF')
            ax.set_xlabel('CV')
            ax.set_title('{:.0f} ns PMF'.format(eval(time[1:]) / 1000), fontsize=20)
            plt.savefig(f'{self.folder}/video/{name}_{i}.png')
            plt.close()
        
        # generate video with ffmpeg

        ffmpeg_command = [ffmpeg,
                '-y',
                '-framerate', '10',
                '-i', f'{name}/{name}_%d.png',
                '-r', '30',
                '-b', '5000k',
                '-vcodec', 'h264',
                f'{name}/{name}.mp4']

        process = subprocess.Popen(ffmpeg_command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=0,
                        cwd=self.folder + '/video/')
        process.stdin.close()

        for line in process.stdout:
            print(line.strip())
        for line in process.stderr:
            print(line.strip())
        
        if remove_img:
            os.system(f'rm {self.folder}/video/{name}_*.png')
    
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
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf

    def generate_pmf_video(self,
                           name,
                           stride=1,
                           remove_img=True,
                           ffmpeg='ffmpeg'):
        """
        Generate a video of the 2D PMF time evolution.

        Parameters
        ----------
        name : str
            The name of the video.
            saved under the folder `video/`.
        stride : int
            The stride of the time series.
        remove_img : bool
            Whether to remove the images after the video is generated.
            Default is True.
        ffmpeg : str
            The path to the ffmpeg executable.
        """
        os.makedirs(self.folder + '/video/', exist_ok=True)
        from tqdm import tqdm
        for i, awh_pmf in tqdm(enumerate(self.awh_results.pmf[::stride]),
                            total=len(self.awh_results.pmf[::stride])):
            time = self.awh_results.timeseries[i*stride]

            awh_cv1 = awh_pmf.T[0][0]
            awh_cv2 = awh_pmf[0].T[1]
            awh_fes = awh_pmf[:,:,2].T

            fig, ax = plt.subplots(figsize=(7,9))
            mappable = ax.contourf(
                        awh_cv1,
                        awh_cv2,
                        awh_fes,
        #                vmax=100,
                        levels=1000)

            ax.set_ylabel(f'$\lambda$')
            ax.set_xlabel('dist (nm)')
            ax.set_title('{:.0f} ns PMF'.format(eval(time[1:]) / 1000), fontsize=20)
            cbar = fig.colorbar(mappable)
            cbar.set_label(f'PMF ({self.unit})')
            plt.savefig(f'{self.folder}/video/{name}_{i}.png')
            plt.close()
        
        # generate video with ffmpeg

        ffmpeg_command = [ffmpeg,
                '-y',
                '-framerate', '10',
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
                        cwd=self.folder + '/video/')
        process.stdin.close()

        for line in process.stdout:
            print(line.strip())
        for line in process.stderr:
            print(line.strip())
        
        if remove_img:
            os.system(f'rm {self.folder}/video/{name}_*.png')

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
        
        return time, awh_pmf_xvg.unit, awh_pmf_xvg.awh_pmf