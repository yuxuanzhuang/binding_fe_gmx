import pandas as pd
import re
import numpy as np

class AWH_LOG(object):
    def __init__(self, location, index=0):
        """
        Read log file from gromacs awh simulation.
        Only lines start with awh will be processed.

        Parameters
        ----------
        location : str
            location of the xvg file
        index : int
            simulation index.
            Default is 0.
        """
        self.filename = location
        self.index = str(index)
        self._read()
        
    def _read(self):
        self._init_data()
        self.covering_times = []
        self.equlibrated_histogram_time = None
        self.out_of_initial_stage_time = None
        for line in self.awh_lines:
            if line.find('equilibrated histogram') != -1:
                self.equlibrated_histogram_time = eval(line.split(' ')[-2])
            elif line.find('covering') != -1:
                self.covering_times.append(eval(line.split(' ')[5]))
            elif line.find('out of the initial stage') != -1:
                self.out_of_initial_stage_time = eval(line.split(' ')[-1][:-2])
            else:
                continue

        self.lambda_components = []
        for line in self.fe_lines:
            lambda_vales = re.findall(r'\d+\.\d+', line)

            self.lambda_components.append(np.asarray([eval(num) for num in lambda_vales]))


    def _init_data(self):
        self.awh_lines = []
        self.fe_lines = []
        skip_lambda = False
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('awh'):
                self.awh_lines.append(line)
            elif line.find('Run time exceeded') != -1:
                skip_lambda = True
            elif line.startswith('Current'):
                if skip_lambda:
                    skip_lambda = False
                    continue
                self.fe_lines.append(line)

    def __repr__(self) -> str:
        return f'AWH_LOG(filename={self.filename}, index={self.index}), covering_times={self.covering_times}, equlibrated_histogram_time={self.equlibrated_histogram_time}, out_of_initial_stage_time={self.out_of_initial_stage_time}'