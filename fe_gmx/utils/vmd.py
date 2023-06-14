import numpy as np


class VMD_object(object):
    """Class to create a VMD object.
    """
    def __init__(self):
        self.vmd_object = self.create_vmd_object()
    
    def create_vmd_object(self):
        """Create a VMD object.
        """
        raise NotImplementedError("create_vmd_object method not implemented.")

    @staticmethod
    def write(vmd_objects, filename, mode='wt'):
        with open(filename, mode) as f:
            f.writelines('draw delete all\n')
            f.writelines('draw materials off\n')
            f.writelines('draw color 15\n')
            for vmd_object in vmd_objects:
                f.write(vmd_object.vmd_object + '\n')


class VMD_line(VMD_object):
    """Class to create a VMD line object.
    """
    def __init__(self, start, end, color='red', width=3):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        super().__init__()

    def create_vmd_object(self):
        """Create a VMD line object.
        """
        vmd_object = f'draw line {{{self.start[0]:.2f} {self.start[1]:.1f} {self.start[2]:.2f}}} '
        vmd_object += f'{{{self.end[0]:.2f} {self.end[1]:.2f} {self.end[2]:.2f}}}'
        vmd_object += f' width {self.width}'

        return vmd_object


class VMD_cylinder(VMD_object):
    """Class to create a VMD cylinder object.
    """
    def __init__(self, start, end, color='red', radius=0.1):
        self.start = start
        self.end = end
        self.color = color
        self.radius = radius
        super().__init__()

    def create_vmd_object(self):
        """Create a VMD cylinder object.
        """
        vmd_object = f'draw cylinder {{{self.start[0]:.2f} {self.start[1]:.1f} {self.start[2]:.2f}}} '
        vmd_object += f'{{{self.end[0]:.2f} {self.end[1]:.2f} {self.end[2]:.2f}}}'
        vmd_object += f' radius {self.radius}'

        return vmd_object


class VMD_triangle(VMD_object):
    """Class to create a VMD triangle object.
    """
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__()

    def create_vmd_object(self):
        """Create a VMD triangle object.
        """
        vmd_object = f'draw triangle {{{self.p1[0]:.2f} {self.p1[1]:.1f} {self.p1[2]:.2f}}} '
        vmd_object += f'{{{self.p2[0]:.2f} {self.p2[1]:.2f} {self.p2[2]:.2f}}}'
        vmd_object += f' {{{self.p3[0]:.2f} {self.p3[1]:.2f} {self.p3[2]:.2f}}}'
#        vmd_object += f' width 0.1'

        return vmd_object


def get_coordinate_triangle(p1, p2, a_p123):
    """
    get the 2D coordinate of p3 based on p1, p2 and angle of p2p1 and p3p1
    """
    a_p123 = np.deg2rad(a_p123)
    p1p2 = p2 - p1
    rot = np.array([[np.cos(a_p123), -np.sin(a_p123)], [np.sin(a_p123), np.cos(a_p123)]])
    p1p3 = np.dot(rot, p1p2)
    p3 = p1 + p1p3
    return p3


def get_coordiante_vector(p1, p2, length):
    """
    get new coordinate of p2 based on p1, p2 and length of p1p2
    """
    p1p2 = p2 - p1
    p1p2_norm = np.linalg.norm(p1p2)
    p1p2_unit = p1p2 / p1p2_norm

    p1p3_norm = length
    p1p3 = p1p2_unit * p1p3_norm

    p3 = p1 + p1p3

    return p3