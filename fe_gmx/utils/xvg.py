import pandas as pd
import numpy as np

class XVG(object):
    def __init__(self, location, index=0, custom_names=None, temp=300):
        """
        Read xvg file from gromacs pull code.

        Parameters
        ----------
        location : str
            location of the xvg file
        index : int
            simulation index.
            Default is 0.
        custom_names : list
            custom names for the pulling dimensions.
            It will be used as the column names of the dataframe.
            If None, the names will be dim_1, dim_2, etc.
            Default is None.
        temp : float
            temperature of the simulation.
        """
        self.filename = location
        self.index = str(index)
        self.custom_names = custom_names
        self.temp = temp
        self._read()
        
    def _read(self):
        self._init_data()
        if self.custom_names is None:
            names = [f'dim_{i+1}' for i in range(self._shape)]
        else:
            names = self.custom_names
        self.data = pd.read_table(self.filename,
                                 header=None,
                                 names=names,
                                 sep='\s+',
                                 on_bad_lines='skip',
                                 skiprows=self._init_line)
        self.data = self.data.iloc[::, :]

        # remove nan lines
        # this is useful when the simulation
        # is still running and the xvg file is not complete
        self.data = self.data.dropna()
        
    def _init_data(self):
        self._init_line = 0
        self._shape = 0
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        # get index of line after lines with @ or #
        for i, line in enumerate(lines):
            if line.startswith('@    xaxis'):
                self.xlabel = line.split('"')[1]
            if line.startswith('@    yaxis'):
                self.ylabel = line.split('"')[1]
                if self.ylabel.find('kJ') != -1:
                    self.unit = 'kJ/mol'
                elif self.ylabel.find('kcal') != -1:
                    self.unit = 'kcal/mol'
                elif self.ylabel.find('(k\\sB\\NT)') != -1:
                    self.unit = 'kT'
                else:
                    self.unit = 'unknown'
            if line.startswith('# AWH metadata: target error'):
                # example # AWH metadata: target error = 4.63 kT = 11.55 kJ/mol
                # get 11.55
                self.taget_error = float(line.split('=')[2].split()[0])
            if line.startswith('# AWH metadata: log sample weight'):
                # example # AWH metadata: log sample weight = 37.92
                # get 37.92
                self.log_sample_weight = float(line.split('=')[1].split()[0])
            elif line.startswith('@') or line.startswith('#'):
                continue
            else:
                self._init_line = i
                line_1 = line
                # estimate the shape of the data
                self._shape = len(line_1.split())
                break

    def __repr__(self) -> str:
        return f'XVG file: {self.filename}, index: {self.index}'

    @property
    def kT(self):
        """
        Energy conversion factor.
        """
        if self.unit == 'kJ/mol':
            return self.temp * 0.00831446261815324
        elif self.unit == 'kcal/mol':
            return self.temp * 0.00198720425864083
        elif self.unit == 'kT':
            return 1
        elif self.unit == 'unknown':
            return 1
        else:
            raise ValueError('Unknown unit.')

            
class AWH_1D_XVG(XVG):
    """
    Special class for AWH xvg file.
    The first column are the pulling dimensions.
    """
    def __init__(self, location, index=0, custom_names=None):
        super().__init__(location, index, custom_names)
        self.convert_1d_array()

    def convert_1d_array(self):
        """
        Convert the data to 2D array.
        """
        
        self.dim1 = self.data.iloc[:,0].unique().shape[0]
        self.awh_pmf = np.zeros((self.dim1, self.data.shape[1]))
    
        for i, arr in enumerate(np.split(self.data, self.dim1)):
            self.awh_pmf[i] = arr

class AWH_2D_XVG(XVG):
    """
    Special class for 2D AWH xvg file.
    The first two columns are the pulling dimensions.
    """
    def __init__(self, location, index=0, custom_names=None):
        super().__init__(location, index, custom_names)
        self.convert_2d_array()

    def convert_2d_array(self):
        """
        Convert the data to 2D array.
        """
        
        self.dim1 = self.data.iloc[:,0].unique().shape[0]
        self.dim2 = self.data.iloc[:,1].unique().shape[0]
        self.awh_pmf = np.zeros((self.dim1, self.dim2, self.data.shape[1]))
    
        for i, arr in enumerate(np.split(self.data, self.dim1)):
            self.awh_pmf[i] = arr

class AWH_3D_XVG(XVG):
    """
    Special class for 3D AWH xvg file.
    The first three columns are the pulling dimensions.
    """
    def __init__(self, location, index=0, custom_names=None):
        super().__init__(location, index, custom_names)
        self.convert_3d_array()

    def convert_3d_array(self):
        """
        Convert the data to 3D array.
        """
        print(self.data.shape)
        self.dim1 = self.data.iloc[:,0].unique().shape[0]
        self.dim2 = self.data.iloc[:,1].unique().shape[0]
        self.dim3 = self.data.iloc[:,2].unique().shape[0]

        self.awh_pmf = np.zeros((self.dim1, self.dim2, self.dim3, self.data.shape[1]))
    
        for i, arrs in enumerate(np.split(self.data, self.dim1)):
            for j, arr in enumerate(np.split(arrs, self.dim2)):
                self.awh_pmf[i, j] = arr    