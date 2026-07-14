#!/usr/bin/env python

#Copyright (c) 2026, Steven Ludtke, Baylor College of Medicine
# Please see the eman3/LICENSE file for licensing information

"""
EMAN3 Transform and Symmetry Classes converted to Python from EMAN2 C++ code.
Partially coded by Qwen 3.5.
"""

import numpy as np
from typing import Dict, List, Optional


class TransformError(Exception):
    """Exception for Transform-related errors"""
    pass

class Transform:
    """
    Transformation class using numpy arrays for all operations.
    
    The transform matrix is a 3x4 numpy array storing rotation (3x3) and 
    translation (3x1). All methods accept/return numpy arrays.
    """
    
    ERR_LIMIT = 1e-6
    
    def __init__(self, dict_params: Optional[Dict] = None):
        """Initialize transform with 3x4 identity matrix"""
        self.matrix: np.ndarray = np.eye(4, dtype=np.float32)[:3, :4].copy()
        if dict_params:
            self.set_params(dict_params)
    
    def to_identity(self) -> None:
        """Set the transform to identity"""
        self.matrix[:] = np.eye(3, dtype=np.float32)
        self.matrix[0, 3] = 0.0
        self.matrix[1, 3] = 0.0
        self.matrix[2, 3] = 0.0
    
    def is_identity(self) -> bool:
        """Check if transform is identity"""
        return np.allclose(self.matrix, np.eye(3, dtype=np.float32), atol=self.ERR_LIMIT)
    
    def is_rot_identity(self) -> bool:
        """Check if rotation part is identity"""
        return np.allclose(self.matrix[:, :3], np.eye(3), atol=self.ERR_LIMIT)
    
    # === Translation methods (numpy arrays) ===
    
    def get_trans(self) -> np.ndarray:
        """Get translation vector as numpy array (3,)"""
        return self.matrix[:, 3].copy()
    
    def get_trans_2d(self) -> np.ndarray:
        """Get 2D translation vector as numpy array (2,)"""
        return self.matrix[:2, 3].copy()
    
    def set_trans(self, trans: np.ndarray) -> None:
        """Set translation from numpy array (3,)"""
        trans = np.asarray(trans, dtype=np.float32)
        self.matrix[0, 3] = trans[0]
        self.matrix[1, 3] = trans[1]
        self.matrix[2, 3] = trans[2]
    
    # === Scale methods ===
    
    def get_scale(self) -> float:
        """Get the scale factor of the transform"""
        det = float(np.linalg.det(self.matrix[:, :3]))
        if det < 0:
            det = -det
        scale = abs(det) ** (1.0 / 3.0)
        int_scale = int(scale)
        if abs(scale - int_scale) < self.ERR_LIMIT:
            scale = float(int_scale)
        return float(scale)
    
    def set_scale(self, scale: float) -> None:
        """Set the scale factor"""
        if scale <= 0:
            raise TransformError("Scale must be positive")
        old_scale = self.get_scale()
        if old_scale == 0:
            raise TransformError("Current transform has zero scale")
        corrected_scale = scale / old_scale
        if not np.isclose(corrected_scale, 1.0, atol=self.ERR_LIMIT):
            self.matrix[:, :3] *= corrected_scale
    
    # === Mirror methods ===
    
    def get_mirror(self) -> bool:
        """Check if transform includes mirror operation"""
        return float(np.linalg.det(self.matrix[:, :3])) < 0
    
    def set_mirror(self, mirror: bool) -> None:
        """Set mirror operation"""
        if self.get_mirror() == mirror:
            return
        self.matrix[0, :] *= -1.0
    
    # === Matrix access ===
    
    def get_matrix(self) -> np.ndarray:
        """Get the 3x4 matrix as numpy array"""
        return self.matrix.copy()
    
    def get_matrix_4x4(self) -> np.ndarray:
        """Get the full 4x4 matrix as numpy array"""
        result = np.eye(4, dtype=np.float32)
        result[:3, :4] = self.matrix
        return result
    
    # === Rotation/Transform composition ===
    
    def rotate(self, by: 'Transform') -> None:
        """Post-multiply: result = by * self"""
        self.matrix[:] = by.matrix @ self.matrix
    
    def rotate_origin(self, by: 'Transform') -> None:
        """Pre-multiply for rotation (rotation only)"""
        self.matrix[:, :3] = by.matrix[:, :3] @ self.matrix[:, :3]
    
    def translate(self, trans: np.ndarray) -> None:
        """Add translation vector to the transform"""
        trans = np.asarray(trans, dtype=np.float32)
        self.matrix[:, 3] += trans
    
    def invert(self) -> 'Transform':
        """Invert the transform in-place"""
        rot = self.matrix[:, :3].copy()
        trans = self.matrix[:, 3].copy()
        det = np.linalg.det(rot)
        if abs(det) < self.ERR_LIMIT:
            raise TransformError("Cannot invert singular transform")
        rot_inv = np.linalg.inv(rot)
        self.matrix[:, :3] = rot_inv
        self.matrix[:, 3] = -rot_inv @ trans
        return self
    
    def inverse(self) -> 'Transform':
        """Return inverted copy of this transform"""
        inv = Transform()
        inv.matrix = self.matrix.copy()
        inv.invert()
        return inv
    
    def transpose_inplace(self) -> None:
        """Transpose the rotation part in-place"""
        self.matrix[:, :3] = self.matrix[:, :3].T
    
    def transpose(self) -> 'Transform':
        """Return transposed copy"""
        t = Transform()
        t.matrix = self.matrix.copy()
        t.transpose_inplace()
        return t
    
    # === Euler rotation methods ===
    
    def set_rotation(self, dict_params: Dict) -> None:
        """Set rotation from dictionary with Euler angle parameters"""
        euler_type = dict_params.get('type', 'eman').lower()
        if euler_type == 'eman':
            self._set_rotation_eman(dict_params)
        elif euler_type == 'spider':
            self._set_rotation_spider(dict_params)
        elif euler_type == 'mrc':
            self._set_rotation_mrc(dict_params)
        elif euler_type == 'quaternion':
            self._set_rotation_quaternion(dict_params)
        elif euler_type == 'matrix':
            self._set_rotation_matrix(dict_params)
        elif euler_type == 'imagic':
            self._set_rotation_imagic(dict_params)
        else:
            raise TransformError(f"Unknown Euler type: {euler_type}")
    
    def _set_rotation_eman(self, dict_params: Dict) -> None:
        az = dict_params.get('az', 0) * np.pi / 180.0
        alt = dict_params.get('alt', 0) * np.pi / 180.0
        phi = dict_params.get('phi', 0) * np.pi / 180.0
        self._set_rotation_from_euler(az, alt, phi)
    
    def _set_rotation_spider(self, dict_params: Dict) -> None:
        phi = (dict_params.get('phi', 0) + 90) * np.pi / 180.0
        theta = dict_params.get('theta', 0) * np.pi / 180.0
        psi = (dict_params.get('psi', 0) - 90) * np.pi / 180.0
        self._set_rotation_from_euler(psi, theta, phi)
    
    def _set_rotation_mrc(self, dict_params: Dict) -> None:
        phi = (dict_params.get('phi', 0) + 90) * np.pi / 180.0
        theta = dict_params.get('theta', 0) * np.pi / 180.0
        omega = (dict_params.get('omega', 0) - 90) * np.pi / 180.0
        self._set_rotation_from_euler(omega, theta, phi)
    
    def _set_rotation_quaternion(self, dict_params: Dict) -> None:
        e0 = dict_params.get('e0', 0)
        e1 = dict_params.get('e1', 0)
        e2 = dict_params.get('e2', 0)
        e3 = dict_params.get('e3', 0)
        self.matrix[0, 0] = e0*e0 + e1*e1 - e2*e2 - e3*e3
        self.matrix[0, 1] = 2.0 * (e1*e2 + e0*e3)
        self.matrix[0, 2] = 2.0 * (e1*e3 - e0*e2)
        self.matrix[1, 0] = 2.0 * (e2*e1 - e0*e3)
        self.matrix[1, 1] = e0*e0 - e1*e1 + e2*e2 - e3*e3
        self.matrix[1, 2] = 2.0 * (e2*e3 + e0*e1)
        self.matrix[2, 0] = 2.0 * (e3*e1 + e0*e2)
        self.matrix[2, 1] = 2.0 * (e3*e2 - e0*e1)
        self.matrix[2, 2] = e0*e0 - e1*e1 - e2*e2 + e3*e3
    
    def _set_rotation_matrix(self, dict_params: Dict) -> None:
        self.matrix[0, 0] = dict_params.get('m11', 1)
        self.matrix[0, 1] = dict_params.get('m12', 0)
        self.matrix[0, 2] = dict_params.get('m13', 0)
        self.matrix[1, 0] = dict_params.get('m21', 0)
        self.matrix[1, 1] = dict_params.get('m22', 1)
        self.matrix[1, 2] = dict_params.get('m23', 0)
        self.matrix[2, 0] = dict_params.get('m31', 0)
        self.matrix[2, 1] = dict_params.get('m32', 0)
        self.matrix[2, 2] = dict_params.get('m33', 1)
    
    def _set_rotation_imagic(self, dict_params: Dict) -> None:
        alpha = dict_params.get('alpha', 0) * np.pi / 180.0
        beta = dict_params.get('beta', 0) * np.pi / 180.0
        gamma = dict_params.get('gamma', 0) * np.pi / 180.0
        self._set_rotation_from_euler(alpha, beta, gamma)
    
    def _set_rotation_from_euler(self, az: float, alt: float, phi: float) -> None:
        """Set rotation from Euler angles in EMAN convention (az, alt, phi)"""
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        caz = np.cos(az)
        saz = np.sin(az)
        callt = np.cos(alt)
        salt = np.sin(alt)
        
        self.matrix[0, 0] = cphi * caz - callt * saz * sphi
        self.matrix[0, 1] = cphi * saz + callt * caz * sphi
        self.matrix[0, 2] = salt * sphi
        self.matrix[1, 0] = -sphi * caz - callt * saz * cphi
        self.matrix[1, 1] = -sphi * saz + callt * caz * cphi
        self.matrix[1, 2] = salt * cphi
        self.matrix[2, 0] = salt * saz
        self.matrix[2, 1] = -salt * caz
        self.matrix[2, 2] = callt
    
    # === Parameter methods ===
    
    def set_params(self, dict_params: Dict) -> None:
        """Set transform parameters from dictionary"""
        if 'type' in dict_params:
            self.set_rotation(dict_params)
        if 'tx' in dict_params:
            self.matrix[0, 3] = float(dict_params['tx'])
        if 'ty' in dict_params:
            self.matrix[1, 3] = float(dict_params['ty'])
        if 'tz' in dict_params:
            self.matrix[2, 3] = float(dict_params['tz'])
        if 'scale' in dict_params:
            self.set_scale(float(dict_params['scale']))
        if 'mirror' in dict_params:
            self.set_mirror(bool(dict_params['mirror']))
    
    def get_rotation(self, euler_type: str = 'eman') -> Dict:
        """Extract rotation parameters in specified Euler convention"""
        euler_type = euler_type.lower()
        scale = self.get_scale()
        if scale == 0:
            raise TransformError("Transform has zero scale")
        
        cosalt = self.matrix[2, 2] / scale
        x_mirror = self.get_mirror()
        x_mirror_scale = -1.0 if x_mirror else 1.0
        
        result = {'type': euler_type}
        
        if euler_type == 'eman':
            if abs(cosalt - 1) < 1e-6:
                alt, az = 0.0, 0.0
                phi = np.arctan2(x_mirror_scale * self.matrix[0, 1], x_mirror_scale * self.matrix[0, 0]) * 180.0 / np.pi
            elif abs(cosalt + 1) < 1e-6:
                alt, az = 180.0, 0.0
                phi = np.arctan2(-x_mirror_scale * self.matrix[0, 1], x_mirror_scale * self.matrix[0, 0]) * 180.0 / np.pi
            else:
                az = np.arctan2(scale * self.matrix[2, 0], -scale * self.matrix[2, 1]) * 180.0 / np.pi
                alt = np.arctan2(np.sqrt(self.matrix[2, 0]**2 + self.matrix[2, 1]**2), abs(self.matrix[2, 2])) * 180.0 / np.pi
                if self.matrix[2, 2] * scale < 0:
                    alt = 180.0 - alt
                phi = np.arctan2(x_mirror_scale * self.matrix[0, 2], self.matrix[1, 2]) * 180.0 / np.pi
            result['az'], result['alt'], result['phi'] = float(az), float(alt), float(phi)
        elif euler_type == 'spider':
            az = np.arctan2(scale * self.matrix[2, 0], -scale * self.matrix[2, 1]) * 180.0 / np.pi
            alt = np.arctan2(np.sqrt(self.matrix[2, 0]**2 + self.matrix[2, 1]**2), abs(self.matrix[2, 2])) * 180.0 / np.pi
            phi = np.arctan2(x_mirror_scale * self.matrix[0, 2], self.matrix[1, 2]) * 180.0 / np.pi
            result['phi'], result['theta'], result['psi'] = float(az - 90.0), float(alt), float(phi + 90.0)
        elif euler_type == 'quaternion':
            trace = self.matrix[0, 0] + self.matrix[1, 1] + self.matrix[2, 2]
            cosomega = (trace - 1.0) / 2.0
            A0 = (self.matrix[1, 2] - self.matrix[2, 1]) / 2.0
            A1 = (self.matrix[2, 0] - self.matrix[0, 2]) / 2.0
            A2 = (self.matrix[0, 1] - self.matrix[1, 0]) / 2.0
            sinomega = np.sqrt(A0**2 + A1**2 + A2**2)
            cosomega = np.clip(cosomega, -1.0, 1.0)
            cosOover2 = np.sqrt((1.0 + cosomega) / 2.0)
            sinOover2 = np.sqrt((1.0 - cosomega) / 2.0)
            n1 = A0 / sinomega if sinomega > 1e-10 else 0
            n2 = A1 / sinomega if sinomega > 1e-10 else 0
            n3 = A2 / sinomega if sinomega > 1e-10 else 0
            result['e0'], result['e1'], result['e2'], result['e3'] = float(cosOover2), float(sinOover2 * n1), float(sinOover2 * n2), float(sinOover2 * n3)
        return result
    
    @staticmethod
    def icos_5_to_2() -> 'Transform':
        """Create transform converting 5-fold to 2-fold in icosahedral symmetry"""
        t = Transform()
        t.set_rotation({'type': 'eman', 'phi': 0, 'az': 90.0, 'alt': 31.717474})
        return t
    
    @staticmethod
    def tet_3_to_2() -> 'Transform':
        """Create transform converting 3-fold to 2-fold in tetrahedral symmetry"""
        t = Transform()
        t.set_rotation({'type': 'eman', 'phi': 45.0, 'az': 0.0, 'alt': 54.73561})
        return t


# === Symmetry3D Classes (numpy arrays) ===


class Symmetry3D:
    """Base class for 3D symmetries"""
    
    def __init__(self):
        self.params: Dict = {}
    
    def get_nsym(self) -> int:
        raise NotImplementedError
    
    def get_sym(self, n: int) -> Transform:
        raise NotImplementedError
    
    def get_delimiters(self, inc_mirror: bool = False) -> Dict:
        raise NotImplementedError
    
    def is_in_asym_unit(self, altitude: float, azimuth: float, inc_mirror: bool = False) -> bool:
        raise NotImplementedError


class CSym(Symmetry3D):
    """Cyclic symmetry"""
    
    def __init__(self, nsym: int = 1):
        super().__init__()
        self.params['nsym'] = nsym
    
    def get_nsym(self) -> int:
        return self.params['nsym']
    
    def get_sym(self, n: int) -> Transform:
        t = Transform()
        t.set_rotation({'type': 'eman', 'az': (n % self.get_nsym()) * 360.0 / self.get_nsym(), 'alt': 0.0, 'phi': 0.0})
        return t
    
    def get_delimiters(self, inc_mirror: bool = False) -> Dict:
        nsym = self.get_nsym()
        return {'alt_max': 180.0 if inc_mirror else 90.0, 'az_max': 360.0 / nsym}
    
    def is_in_asym_unit(self, altitude: float, azimuth: float, inc_mirror: bool = False) -> bool:
        d = self.get_delimiters(inc_mirror)
        if self.get_nsym() != 1 and azimuth < 0:
            return False
        return altitude <= d['alt_max'] and azimuth < d['az_max']
    
    def get_name(self) -> str:
        return 'c'
    
    def get_desc(self) -> str:
        return f'C{self.get_nsym()} symmetry'


class DSym(Symmetry3D):
    """Dihedral symmetry"""
    
    def __init__(self, nsym: int = 1):
        super().__init__()
        self.params['nsym'] = nsym
    
    def get_nsym(self) -> int:
        return 2 * self.params['nsym']
    
    def get_sym(self, n: int) -> Transform:
        nsym = self.get_nsym()
        t = Transform()
        if n >= nsym // 2:
            t.set_rotation({'type': 'eman', 'az': ((n % nsym) - nsym // 2) * 360.0 / (nsym // 2), 'alt': 180.0, 'phi': 0.0})
        else:
            t.set_rotation({'type': 'eman', 'az': (n % nsym) * 360.0 / (nsym // 2), 'alt': 0.0, 'phi': 0.0})
        return t
    
    def get_delimiters(self, inc_mirror: bool = False) -> Dict:
        return {'alt_max': 90.0, 'az_max': 360.0 / self.get_nsym() if inc_mirror else 180.0 / self.get_nsym()}
    
    def is_in_asym_unit(self, altitude: float, azimuth: float, inc_mirror: bool = False) -> bool:
        d = self.get_delimiters(inc_mirror)
        if self.get_nsym() == 1 and inc_mirror:
            return 0 <= altitude <= d['alt_max'] and 0 <= azimuth < d['az_max']
        return 0 <= altitude <= d['alt_max'] and 0 <= azimuth < d['az_max']
    
    def get_name(self) -> str:
        return 'd'
    
    def get_desc(self) -> str:
        return f'D{self.get_nsym()} symmetry'


class HSym(Symmetry3D):
    """Helical symmetry"""
    
    def __init__(self, nsym: int = 1, daz: float = 0.0, tz: float = 0.0, maxtilt: float = 90.0, nstart: int = 1, apix: float = 1.0):
        super().__init__()
        self.params = {'nsym': nsym, 'daz': daz, 'tz': tz, 'maxtilt': maxtilt, 'nstart': nstart, 'apix': apix}
    
    def get_nsym(self) -> int:
        return self.params['nsym']
    
    def get_sym(self, n: int) -> Transform:
        daz = self.params['daz']
        nstart = self.params['nstart']
        tz = self.params['tz'] / self.params.get('apix', 1.0)
        t = Transform()
        ii = (n + 1) // 2
        if n > 1 and n % 2 == 0:
            ii *= -1
        az = (ii % nstart) * (360.0 / nstart) + (ii // nstart) * daz
        t.set_rotation({'type': 'eman', 'az': az, 'alt': 0.0, 'phi': 0.0})
        t.set_trans(np.array([0.0, 0.0, (ii // nstart) * tz], dtype=np.float32))
        return t
    
    def get_delimiters(self, inc_mirror: bool = False) -> Dict:
        maxtilt = self.params.get('maxtilt', 90.0)
        return {'alt_max': 90.0, 'alt_min': 90.0 - maxtilt, 'az_max': 360.0}
    
    def is_in_asym_unit(self, altitude: float, azimuth: float, inc_mirror: bool = False) -> bool:
        d = self.get_delimiters(inc_mirror)
        alt_max, alt_min = d['alt_max'], d['alt_min']
        if inc_mirror:
            alt_min -= self.params.get('maxtilt', 90.0)
        return alt_min <= altitude <= alt_max and 0 <= azimuth <= d['az_max']
    
    def get_name(self) -> str:
        return 'h'
    
    def get_desc(self) -> str:
        return 'Helical symmetry'


class PlatonicSym(Symmetry3D):
    """Base class for Platonic solid symmetries"""
    
    def __init__(self):
        super().__init__()
        self.platonic_params: Dict = {}
        self.init()
    
    def init(self) -> None:
        cap_sig = 2.0 * np.pi / self.get_max_csym()
        self.platonic_params['az_max'] = cap_sig
        self.platonic_params['alt_max'] = np.arccos(1.0 / (np.sqrt(3.0) * np.tan(cap_sig / 2.0)))
        self.platonic_params['theta_c_on_two'] = 0.5 * np.arccos(np.cos(cap_sig) / (1.0 - np.cos(cap_sig)))
    
    def get_max_csym(self) -> int:
        raise NotImplementedError
    
    def get_az_alignment_offset(self) -> float:
        return 0.0
    
    def is_in_asym_unit(self, altitude: float, azimuth: float, inc_mirror: bool = False) -> bool:
        d = self.get_delimiters(inc_mirror)
        az_max, alt_max = d['az_max'], d['alt_max']
        if 0 <= altitude <= alt_max and 0 <= azimuth < az_max:
            tmpaz = (azimuth * np.pi / 180.0)
            cap_sig = self.platonic_params['az_max']
            alt_max_rad = self.platonic_params['alt_max']
            tmpaz = min(tmpaz, cap_sig - tmpaz)
            tmpalt = altitude * np.pi / 180.0
            lower_bound = self.platonic_alt_lower_bound(tmpaz, alt_max_rad)
            return lower_bound > tmpalt
        return False
    
    def platonic_alt_lower_bound(self, azimuth: float, alpha: float) -> float:
        cap_sig = self.platonic_params['az_max']
        theta_c_on_two = self.platonic_params['theta_c_on_two']
        baldwin_lower = np.sin(cap_sig / 2.0 - azimuth) / np.tan(theta_c_on_two)
        baldwin_lower += np.sin(azimuth) / np.tan(alpha)
        baldwin_lower /= np.sin(cap_sig / 2.0)
        return float(np.arctan(1.0 / baldwin_lower))
    
    def get_delimiters(self, inc_mirror: bool = False) -> Dict:
        ret = {
            'az_max': float(self.platonic_params['az_max'] * 180.0 / np.pi),
            'alt_max': float(self.platonic_params['alt_max'] * 180.0 / np.pi)
        }
        if not inc_mirror and self.get_name() in ['icos', 'oct']:
            ret['az_max'] *= 0.5
        return ret
    
    def get_nsym(self) -> int:
        raise NotImplementedError
    
    def get_sym(self, n: int) -> Transform:
        raise NotImplementedError
    
    def is_platonic_sym(self) -> bool:
        return True


class IcosahedralSym(PlatonicSym):
    """Icosahedral symmetry (60 operations)"""
    
    def __init__(self):
        super().__init__()
    
    def get_max_csym(self) -> int:
        return 5
    
    def get_nsym(self) -> int:
        return 60
    
    def get_sym(self, n: int) -> Transform:
        idx = n % 60
        ICOS = [
            (0, 0, 0), (0, 0, 288), (0, 0, 216), (0, 0, 144), (0, 0, 72),
            (0, 63.4349, 36), (0, 63.4349, 324), (0, 63.4349, 252), (0, 63.4349, 180), (0, 63.4349, 108),
            (72, 63.4349, 36), (72, 63.4349, 324), (72, 63.4349, 252), (72, 63.4349, 180), (72, 63.4349, 108),
            (144, 63.4349, 36), (144, 63.4349, 324), (144, 63.4349, 252), (144, 63.4349, 180), (144, 63.4349, 108),
            (216, 63.4349, 36), (216, 63.4349, 324), (216, 63.4349, 252), (216, 63.4349, 180), (216, 63.4349, 108),
            (288, 63.4349, 36), (288, 63.4349, 324), (288, 63.4349, 252), (288, 63.4349, 180), (288, 63.4349, 108),
            (36, 116.5651, 0), (36, 116.5651, 288), (36, 116.5651, 216), (36, 116.5651, 144), (36, 116.5651, 72),
            (108, 116.5651, 0), (108, 116.5651, 288), (108, 116.5651, 216), (108, 116.5651, 144), (108, 116.5651, 72),
            (180, 116.5651, 0), (180, 116.5651, 288), (180, 116.5651, 216), (180, 116.5651, 144), (180, 116.5651, 72),
            (252, 116.5651, 0), (252, 116.5651, 288), (252, 116.5651, 216), (252, 116.5651, 144), (252, 116.5651, 72),
            (324, 116.5651, 0), (324, 116.5651, 288), (324, 116.5651, 216), (324, 116.5651, 144), (324, 116.5651, 72),
            (0, 180, 0), (0, 180, 288), (0, 180, 216), (0, 180, 144), (0, 180, 72)
        ]
        az, alt, phi = ICOS[idx]
        t = Transform()
        t.set_rotation({'type': 'eman', 'az': az, 'alt': alt, 'phi': phi})
        return t
    
    def get_az_alignment_offset(self) -> float:
        return 234.0
    
    def get_name(self) -> str:
        return 'icos'
    
    def get_desc(self) -> str:
        return 'Icosahedral symmetry (60-fold)'


class TetrahedralSym(PlatonicSym):
    """Tetrahedral symmetry (12 operations)"""
    
    def __init__(self):
        super().__init__()
    
    def get_max_csym(self) -> int:
        return 3
    
    def get_nsym(self) -> int:
        return 12
    
    def get_sym(self, n: int) -> Transform:
        idx = n % 12
        TET = [
            (0, 0, 0), (0, 0, 120), (0, 0, 240),
            (0, 54.7356, 60), (0, 54.7356, 180), (0, 54.7356, 300),
            (120, 54.7356, 60), (120, 54.7356, 180), (120, 54.7356, 300),
            (240, 54.7356, 60), (240, 54.7356, 180), (240, 54.7356, 300)
        ]
        az, alt, phi = TET[idx]
        t = Transform()
        t.set_rotation({'type': 'eman', 'az': az, 'alt': alt, 'phi': phi})
        return t
    
    def get_name(self) -> str:
        return 'tet'
    
    def get_desc(self) -> str:
        return 'Tetrahedral symmetry (12-fold)'


class OctahedralSym(PlatonicSym):
    """Octahedral symmetry (24 operations)"""
    
    def __init__(self):
        super().__init__()
    
    def get_max_csym(self) -> int:
        return 4
    
    def get_nsym(self) -> int:
        return 24
    
    def get_sym(self, n: int) -> Transform:
        idx = n % 24
        OCT = [
            (0, 0, 0), (0, 0, 90), (0, 0, 180), (0, 0, 270),
            (0, 90, 0), (0, 90, 90), (0, 90, 180), (0, 90, 270),
            (90, 90, 0), (90, 90, 90), (90, 90, 180), (90, 90, 270),
            (180, 90, 0), (180, 90, 90), (180, 90, 180), (180, 90, 270),
            (270, 90, 0), (270, 90, 90), (270, 90, 180), (270, 90, 270),
            (0, 180, 0), (0, 180, 90), (0, 180, 180), (0, 180, 270)
        ]
        az, alt, phi = OCT[idx]
        t = Transform()
        t.set_rotation({'type': 'eman', 'az': az, 'alt': alt, 'phi': phi})
        return t
    
    def get_name(self) -> str:
        return 'oct'
    
    def get_desc(self) -> str:
        return 'Octahedral symmetry (24-fold)'


class Icosahedral2Sym(PlatonicSym):
    """Icosahedral symmetry using full 3x3 matrices (60 operations)"""
    
    def __init__(self):
        super().__init__()
        self._matrices = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0.30902, -0.80902, 0.5, 0.80902, 0.5, 0.30902, -0.5, 0.30902, 0.80902],
            [-0.80902, -0.5, 0.30902, 0.5, -0.30902, 0.80902, -0.30902, 0.80902, 0.5],
            [-0.80902, 0.5, -0.30902, -0.5, -0.30902, 0.80902, 0.30902, 0.80902, 0.5],
            [0.30902, 0.80902, -0.5, -0.80902, 0.5, 0.30902, 0.5, 0.30902, 0.80902],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        ], dtype=np.float32)
    
    def get_max_csym(self) -> int:
        return 5
    
    def get_nsym(self) -> int:
        return 60
    
    def get_sym(self, n: int) -> Transform:
        idx = n % 60
        t = Transform()
        matrix_flat = self._matrices[idx]
        t.matrix[:, 0] = matrix_flat[0:3]
        t.matrix[:, 1] = matrix_flat[3:6]
        t.matrix[:, 2] = matrix_flat[6:9]
        return t
    
    def get_az_alignment_offset(self) -> float:
        return 234.0
    
    def get_name(self) -> str:
        return 'icos2'
    
    def get_desc(self) -> str:
        return 'Icosahedral symmetry (60-fold, matrix-based)'


def get_symmetry(sym_name: str, **kwargs) -> Symmetry3D:
    """Factory function to create symmetry objects"""
    sym_name = sym_name.lower()
    if sym_name == 'c':
        return CSym(nsym=kwargs.get('nsym', 1))
    elif sym_name == 'd':
        return DSym(nsym=kwargs.get('nsym', 1))
    elif sym_name == 'h':
        return HSym(nsym=kwargs.get('nsym', 1), daz=kwargs.get('daz', 0.0), tz=kwargs.get('tz', 0.0), maxtilt=kwargs.get('maxtilt', 90.0), nstart=kwargs.get('nstart', 1))
    elif sym_name == 'tet':
        return TetrahedralSym()
    elif sym_name == 'oct':
        return OctahedralSym()
    elif sym_name == 'icos':
        return IcosahedralSym()
    elif sym_name == 'icos2':
        return Icosahedral2Sym()
    else:
        raise TransformError(f"Unknown symmetry type: {sym_name}")
