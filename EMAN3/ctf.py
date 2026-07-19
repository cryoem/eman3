#!/usr/bin/env python

#Copyright (c) 2026, Steven Ludtke, Baylor College of Medicine
# Please see the eman3/LICENSE file for licensing information

"""
EMAN2Ctf class converted from EMAN2 C++ code in libEM/ctf.{h,cpp}.
Additional CTF classes will be added to this file later.
Partially coded by Qwen 3.6.
"""

import math
import re
import numpy as np


class CtfTypeError(Exception):
	"""Exception for CTF type errors"""
	pass


# Contrast Transfer Function computation types, from base Ctf class
class CtfType:
	CTF_AMP = 0				# ctf amplitude only
	CTF_SIGN = 1			# ctf sign (+-1)
	CTF_BACKGROUND = 2		# Background, no ctf oscillation
	CTF_SNR = 3			# Signal to noise ratio
	CTF_SNR_SMOOTH = 4		# Signal to noise ratio, smoothed
	CTF_WIENER_FILTER = 5	# Wiener Filter = 1/(1+1/snr)
	CTF_TOTAL = 6			# AMP*AMP+NOISE
	CTF_FITREF = 7			# CTF amplitude squared without B-factor and low res zeroed
	CTF_NOISERATIO = 8		# 1-Noise/Total
	CTF_INTEN = 9			# ctf intensity only (no background or envelope)
	CTF_POWEVAL = 10		# ctf intensity, no B-factor and filtered below 20 A
	CTF_ALIFILT = 11		# ctf intensity -> mean subtracted for phase-flipping near zeroes
	CTF_ABS = 12			# ctf amplitude only (abs value)


class EMAN2Ctf:
	"""EMAN2-style CTF model.

	Contrast Transfer Function (CTF) describes the transfer of information
	from the object to the contrast observed in the image for electron microscopy.

	Inherits conceptual base attributes from Ctf:
		defocus  - Defocus in microns (positive = underfocus)
		bfactor  - B-factor in A^2, x-ray convention (e^-B/4 s^2 in amplitude space)
		voltage  - Microscope voltage in kV
		cs       - Spherical aberration coefficient in mm
		apix     - Angstroms per pixel

	EMAN2-specific attributes:
		dfdiff   - Defocus difference for astigmatism (defocus is major elliptical axis)
		dfang    - Angle of major elliptical axis in degrees, counterclockwise from x
		ampcont  - Amplitude contrast as a percentage (e.g. 10 for 10%)
		dsbg     - ds value for background and SNR
		background - Background intensity, 1 value per radial pixel
		snr      - SNR, 1 value per radial pixel
	"""

	def __init__(self):
		self.defocus = 0.0
		self.dfdiff = 0.0
		self.dfang = 0.0
		self.bfactor = 0.0
		self.ampcont = 0.0
		self.voltage = 0.0
		self.cs = 0.0
		self.apix = 1.0
		self.dsbg = -1.0
		self.background = []
		self.snr = []

	def wavelength(self):
		"""Electron wavelength in Angstroms (relativistic)."""
		lam = 12.2639 / math.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)
		return lam

	def defocus_at_angle(self, ang):
		"""Returns defocus as a function of angle.

		ang - angle in radians
		"""
		if self.dfdiff == 0.0:
			return self.defocus
		return self.defocus + self.dfdiff / 2.0 * math.cos(2.0 * ang - 2.0 * math.pi / 180.0 * self.dfang)

	def get_defocus(self):
		"""Returns the defocus in microns."""
		return self.defocus

	def get_dfdiff(self):
		"""Returns the defocus difference for astigmatism."""
		return self.dfdiff

	def get_bfactor(self):
		"""Returns the B-factor in A^2."""
		return self.bfactor

	def get_phase(self):
		"""Returns the phase shift based on amplitude contrast.

		Phase is defined as zero for pure phase contrast (user convenience).
		The phase shift used in cos(gamma - phase) is shifted -pi/2 from this.
		"""
		if self.ampcont > -100.0 and self.ampcont <= 100.0:
			return math.asin(self.ampcont / 100.0)
		if self.ampcont > 100.0:
			return math.pi - math.asin(2.0 - self.ampcont / 100.0)
		return -math.pi - math.asin(-2.0 - self.ampcont / 100.0)

	def set_phase(self, phase):
		"""Sets the amplitude contrast depending on the given phase shift (radians)."""
		if phase >= -math.pi / 2.0 and phase < math.pi / 2.0:
			self.ampcont = math.sin(phase) * 100.0
		elif phase >= math.pi / 2.0 and phase <= math.pi:
			self.ampcont = (2.0 - math.sin(phase)) * 100.0
		elif phase > -math.pi and phase < -math.pi / 2.0:
			self.ampcont = (-2.0 - math.sin(phase)) * 100.0

	def calc_noise(self, s):
		"""Calculate noise at spatial frequency s using background lookup."""
		si = int(s / self.dsbg)
		bg_len = len(self.background)
		if si >= bg_len or si < 0:
			return self.background[-1] if self.background else 0.0
		return self.background[si]

	def to_dict(self):
		"""Returns a dictionary representation of the CTF parameters."""
		d = {}
		d["defocus"] = self.defocus
		d["dfdiff"] = self.dfdiff
		d["dfang"] = self.dfang
		d["bfactor"] = self.bfactor
		d["ampcont"] = self.ampcont
		d["voltage"] = self.voltage
		d["cs"] = self.cs
		d["apix"] = self.apix
		d["dsbg"] = self.dsbg
		d["background"] = list(self.background)
		d["snr"] = list(self.snr)
		return d

	def from_dict(self, d):
		"""Initialize CTF parameters from a dictionary."""
		self.defocus = float(d.get("defocus", 0.0))
		self.dfdiff = float(d.get("dfdiff", 0.0))
		self.dfang = float(d.get("dfang", 0.0))
		self.bfactor = float(d.get("bfactor", 0.0))
		self.ampcont = float(d.get("ampcont", 0.0))
		self.voltage = float(d.get("voltage", 0.0))
		self.cs = float(d.get("cs", 0.0))
		self.apix = float(d.get("apix", 1.0))
		self.dsbg = float(d.get("dsbg", -1.0))
		if "background" in d:
			self.background = list(d["background"])
		else:
			self.background = []
		if "snr" in d:
			self.snr = list(d["snr"])
		else:
			self.snr = []

	def to_vector(self):
		"""Returns a flat list of all CTF parameters for serialization.

		Order: [defocus, dfdiff, dfang, bfactor, ampcont, voltage, cs, apix, dsbg,
		         bg_len, <background values>, snr_len, <snr values>]
		"""
		v = []
		v.append(self.defocus)
		v.append(self.dfdiff)
		v.append(self.dfang)
		v.append(self.bfactor)
		v.append(self.ampcont)
		v.append(self.voltage)
		val = self.cs
		v.append(val)
		v.append(self.apix)
		v.append(self.dsbg)
		bg_len = len(self.background)
		v.append(float(bg_len))
		for val in self.background:
			v.append(val)
		snr_len = len(self.snr)
		v.append(float(snr_len))
		for val in self.snr:
			v.append(val)
		return v

	def from_vector(self, vctf):
		"""Initialize CTF parameters from a flat list.

		Order matches to_vector().
		"""
		self.defocus = float(vctf[0])
		self.dfdiff = float(vctf[1])
		self.dfang = float(vctf[2])
		self.bfactor = float(vctf[3])
		self.ampcont = float(vctf[4])
		self.voltage = float(vctf[5])
		self.cs = float(vctf[6])
		self.apix = float(vctf[7])
		self.dsbg = float(vctf[8])
		bg_len = int(vctf[9])
		self.background = [float(vctf[i + 10]) for i in range(bg_len)]
		end_bg = 10 + bg_len
		snr_len = int(vctf[end_bg])
		self.snr = [float(vctf[end_bg + 1 + j]) for j in range(snr_len)]

	def to_string(self):
		"""Returns a compact string representation of the CTF.

		Format: E<defocus> <dfdiff> <dfang> <bfactor> <ampcont> <voltage> <cs>
		        <apix> <dsbg> <bg_count>,<bg_val1>,<bg_val2>,... <snr_count>,<snr_val1>,...
		The leading 'E' identifies this as an EMAN2 CTF string.
		"""
		header_vals = [
			f"{self.defocus:.10g}",
			f"{self.dfdiff:.10g}",
			f"{self.dfang:.10g}",
			f"{self.bfactor:.10g}",
			f"{self.ampcont:.10g}",
			f"{self.voltage:.10g}",
			f"{self.cs:.10g}",
			f"{self.apix:.10g}",
			f"{self.dsbg:.10g}",
			str(len(self.background)),
		]

		ret = "E" + " ".join(header_vals)

		for val in self.background:
			ret += f",{val:.3g}"

		ret += f" {len(self.snr)}"
		for val in self.snr:
			ret += f",{val:.3g}"

		return ret

	def from_string(self, ctf_str):
		"""Parse a CTF string and initialize parameters.

		String format:
		  E<defocus> <dfdiff> <dfang> <bfactor> <ampcont> <voltage> <cs> <apix> <dsbg> <bg_len>,<bg1>,<bg2>,... <snr_len>,<snr1>,<snr2>,...

		Returns 0 on success, 1 on parse error.
		Raises CtfTypeError if the string is empty or doesn't start with 'E'.
		"""
		if not ctf_str:
			raise CtfTypeError("Empty CTF string")

		ctf_str = ctf_str.strip()
		if ctf_str[0] != 'E':
			raise CtfTypeError(f"Trying to initialize CTF object with bad string (type='{ctf_str[0]}')")

		try:
			# Match: E followed by 8 floats, 1 float (dsbg), and 1 int (bg_len)
			# C++ format: E immediately followed by first value, spaces between rest
			header_pattern = r'E\s*(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(\d+)'
			match = re.match(header_pattern, ctf_str)
			if not match:
				return 1

			self.defocus = float(match.group(1))
			self.dfdiff = float(match.group(2))
			self.dfang = float(match.group(3))
			self.bfactor = float(match.group(4))
			self.ampcont = float(match.group(5))
			self.voltage = float(match.group(6))
			self.cs = float(match.group(7))
			self.apix = float(match.group(8))
			self.dsbg = float(match.group(9))
			# bg_len is match.group(10) but not stored separately

			# Everything after the header match may be: ,val,val,... snr_len,val,val,...
			remainder = ctf_str[match.end():]

			# Split remainder into background array part and SNR part
			# Background values are comma-separated immediately after bg_len count
			# They may include the trailing part up to a space+digit (the snr_len)
			raw_after = remainder.lstrip()

			self.background = []
			self.snr = []

			# Find where the SNR section starts: a standalone integer after a space
			snr_match = re.search(r'\s(\d+)(?:,|$)', raw_after)
			if snr_match:
				bg_raw = raw_after[:snr_match.start()].strip()
				snr_raw = raw_after[snr_match.start():].strip()
			else:
				# No SNR section found
				bg_raw = raw_after.strip()
				snr_raw = ''

			# Parse background comma-separated values (may be empty)
			if bg_raw.startswith(','):
				self.background = [float(x) for x in bg_raw[1:].split(',') if x]

			# Parse SNR: "snr_len,val1,val2,..." or "snr_len"
			if snr_raw:
				snr_parts = snr_raw.split(',')
				try:
					snr_count = int(snr_parts[0])
					self.snr = [float(x) for x in snr_parts[1:snr_count+1] if x]
				except (ValueError, IndexError):
					pass

		except (ValueError, IndexError):
			return 1

		return 0

	def copy_from(self, other_ctf):
		"""Copy parameters from another EMAN2Ctf object."""
		if other_ctf:
			self.defocus = other_ctf.defocus
			self.dfdiff = other_ctf.dfdiff
			self.dfang = other_ctf.dfang
			self.bfactor = other_ctf.bfactor
			self.ampcont = other_ctf.ampcont
			self.voltage = other_ctf.voltage
			self.cs = other_ctf.cs
			self.apix = other_ctf.apix
			self.dsbg = other_ctf.dsbg
			self.background = list(other_ctf.background)
			self.snr = list(other_ctf.snr)

	def equal(self, other):
		"""Check if two CTF objects have the same parameters."""
		if not isinstance(other, EMAN2Ctf):
			return False
		if self.defocus != other.defocus:
			return False
		if self.dfdiff != other.dfdiff:
			return False
		if self.dfang != other.dfang:
			return False
		if self.bfactor != other.bfactor:
			return False
		if self.ampcont != other.ampcont:
			return False
		if self.voltage != other.voltage:
			return False
		if self.cs != other.cs:
			return False
		if self.apix != other.apix:
			return False
		if self.dsbg != other.dsbg:
			return False
		if len(self.background) != len(other.background):
			return False
		for i in range(len(self.background)):
			if self.background[i] != other.background[i]:
				return False
		if len(self.snr) != len(other.snr):
			return False
		for i in range(len(self.snr)):
			if self.snr[i] != other.snr[i]:
				return False
		return True

	def zero(self, n=0):
		"""Calculate the CTF zero crossings.

		n - specific zero index, or 0 for all zeros.
		Returns a list of spatial frequencies at which CTF crosses zero.
		"""
		lam = self.wavelength()
		ac = self.ampcont / 100.0
		phase_shift = math.asin(ac)

		zeros = []
		num_zeros = 50
		for m in range(1, num_zeros):
			gamma_target = (math.pi * (m - 0.5)) + phase_shift

			r1_sq = gamma_target / (-2.0 * math.pi * self.defocus * lam)
			if r1_sq < 0:
				continue
			r1 = math.sqrt(abs(r1_sq)) * 1e-4

			r2_sq = gamma_target / (2.0 * math.pi * 2.5e6 * self.cs * lam * lam * lam)

			zeros.append(float(r1))
			if r2_sq > 0:
				r2 = math.sqrt(r2_sq) ** (1.0 / 3.0)
				if r2 < 1.0:
					zeros.append(float(r2))

		if n == 0:
			return sorted(set(zeros))
		if 0 <= n < len(zeros):
			return zeros[n]
		raise CtfTypeError(f"Zero index {n} out of range ({len(zeros)} zeros found)")

	def _interpolate_background(self, s):
		"""Interpolate background noise at given spatial frequencies (1D)."""
		if not self.background or self.dsbg <= 0:
			return np.zeros_like(s)
		noise_vals = np.array(self.background, dtype=np.float32)
		f = s / self.dsbg
		j = np.floor(f).astype(int)
		frac = f - j
		j_clipped = np.clip(j, 0, len(noise_vals) - 2)
		return noise_vals[j_clipped] * (1.0 - frac) + noise_vals[j_clipped + 1] * frac

	def _interpolate_background_2d(self, R):
		"""Interpolate background noise for a 2D radial frequency grid."""
		if not self.background or self.dsbg <= 0:
			return np.zeros_like(R)
		noise_vals = np.array(self.background, dtype=np.float32)
		f = R / self.dsbg
		j = np.floor(f).astype(int)
		frac = f - j
		j_clipped = np.clip(j, 0, len(noise_vals) - 2)
		return noise_vals[j_clipped] * (1.0 - frac) + noise_vals[j_clipped + 1] * frac

	def compute_1d(self, size, ds, ctf_type, struct_factor=None):
		"""Compute a 1D CTF curve.

		size - image dimension (output array is size//2 elements)
		ds - spatial frequency step (typically 1/(apix * size))
		ctf_type - one of the CtfType constants
		struct_factor - optional numpy array for structure factor weighting

		Returns a numpy float32 array.
		"""
		if self.voltage <= 0:
			raise CtfTypeError("Voltage must be set (voltage > 0)")

		npts = size // 2
		result = np.zeros(npts, dtype=np.float32)

		s = np.arange(npts, dtype=np.float32) * ds
		s_sq = s * s

		lam = self.wavelength()
		g1 = math.pi / 2.0 * self.cs * 1e7 * lam ** 3
		g2_coeff = math.pi * lam * 1e4
		acac = math.pi / 2.0 - self.get_phase()

		gam = -g1 * s_sq ** 2 + g2_coeff * self.defocus * s_sq
		bf_env = np.exp(-(self.bfactor / 4.0) * s_sq)

		if ctf_type == CtfType.CTF_AMP:
			result = np.cos(gam - acac) * bf_env

		elif ctf_type == CtfType.CTF_ABS:
			result = np.abs(np.cos(gam - acac) * bf_env)

		elif ctf_type == CtfType.CTF_INTEN:
			val = np.cos(gam - acac) * bf_env
			result = val * val

		if ctf_type == CtfType.CTF_SIGN:
			v = np.cos(gam - acac)
			result = np.where(v < 0, -1.0, 1.0).astype(np.float32)

		elif ctf_type == CtfType.CTF_BACKGROUND:
			if self.background and self.dsbg > 0:
				result = self._interpolate_background(s)

		elif ctf_type == CtfType.CTF_SNR:
			if not self.snr:
				raise CtfTypeError("CTF_SNR requires SNR data to be set")
			snr_vals = np.array(self.snr, dtype=np.float32)
			f = s / self.dsbg
			j = np.floor(f).astype(int)
			frac = f - j
			j_clipped = np.clip(j, 0, len(snr_vals) - 2)
			result = (snr_vals[j_clipped] * (1.0 - frac) + snr_vals[j_clipped + 1] * frac).astype(np.float32)

		elif ctf_type == CtfType.CTF_SNR_SMOOTH:
			if not self.snr or not self.background:
				raise CtfTypeError("CTF_SNR_SMOOTH requires SNR and background data")
			snr_vals = np.array(self.snr, dtype=np.float32)

			tsnr = (np.cos(gam - acac) * bf_env) ** 2 / np.maximum(self._interpolate_background(s), 0.001)

			f_bg = s / self.dsbg
			j_bg = np.floor(f_bg).astype(int)
			frac_bg = f_bg - j_bg
			j_clipped = np.clip(j_bg, 0, len(snr_vals) - 2)
			dsnr = (snr_vals[j_clipped] * (1.0 - frac_bg) + snr_vals[j_clipped + 1] * frac_bg).astype(np.float32)

			npsm = max(npts // 25, 2)
			result[0] = 0.0
			for i in range(1, npts):
				lo = max(i - npsm, 1)
				hi = min(i + npsm, npts - 1)
				x = tsnr[lo:hi+1]
				y = dsnr[lo:hi+1]
				n_pts = len(x)
				sum_x = np.sum(x)
				sum_y = np.sum(y)
				sum_xx = np.sum(x * x)
				sum_xy = np.sum(x * y)
				div = n_pts * sum_xx - sum_x * sum_x
				if div != 0:
					slope = (n_pts * sum_xy - sum_x * sum_y) / div
					result[i] = slope * tsnr[i]
				else:
					result[i] = 0.0
				if result[i] < 0:
					result[i] = 0.0

		elif ctf_type == CtfType.CTF_WIENER_FILTER:
			if not self.snr:
				raise CtfTypeError("CTF_WIENER_FILTER requires SNR data")
			snr_vals = np.array(self.snr, dtype=np.float32)
			f = s / self.dsbg
			j = np.floor(f).astype(int)
			frac = f - j
			j_clipped = np.clip(j, 0, len(snr_vals) - 2)
			result = (snr_vals[j_clipped] * (1.0 - frac) + snr_vals[j_clipped + 1] * frac).astype(np.float32)
			result[result < 0] = 0.0
			result = result / (result + 1.0)
			result[0] = 0.0

		elif ctf_type == CtfType.CTF_TOTAL:
			val = np.cos(gam - acac) * bf_env
			signal = val * val
			if struct_factor is not None:
				result = (signal * struct_factor + self._interpolate_background(s)).astype(np.float32)
			else:
				result = (signal + self._interpolate_background(s)).astype(np.float32)

		elif ctf_type == CtfType.CTF_NOISERATIO:
			val = np.cos(gam - acac) * bf_env
			signal = val * val
			if struct_factor is not None:
				signal = signal * struct_factor
			noise = self._interpolate_background(s)
			result = (signal / (signal + noise)).astype(np.float32)

		return result.astype(np.float32)

	def compute_1d_fromimage(self, size, ds, image_complex):
		"""Compute a 1D power spectrum from a complex FFT image.

		size - image dimension
		ds - spatial frequency step
		image_complex - numpy complex64/128 array of shape (ny, nx)
		        with Fourier data in standard fft2 layout.

		Compensates for astigmatism by binning using angle-dependent radial
		frequency. Returns a numpy float32 array of size//2.
		"""
		npts = size // 2
		ret = np.zeros(npts, dtype=np.float32)
		norm = np.zeros(npts, dtype=np.float32)

		ny, nx = image_complex.shape

		x_arr = np.arange(nx // 2)
		y_arr = np.arange(-ny // 2, ny // 2)
		X, Y = np.meshgrid(x_arr, y_arr)

		r_base = np.hypot(X, Y) * ds
		a_ang = np.arctan2(Y, X)

		df_adj = self.defocus + self.dfdiff / 2.0 * np.cos(
			2.0 * a_ang - 2.0 * math.pi / 180.0 * self.dfang
		)

		lam = self.wavelength()
		df_sq_scaled = df_adj ** 2 * 1e8
		term = -2e11 * self.cs * r_base ** 2 * df_adj * lam ** 2
		term4 = 1e14 * self.cs ** 2 * (r_base * lam) ** 4
		disc = np.sqrt(np.maximum(df_sq_scaled + term + term4, 0))
		numer = df_adj * 1e4 - disc
		denom = 1e7 * self.cs * lam ** 2
		s_new = np.sqrt(np.maximum(numer / denom, 0)) / ds

		intensity = np.abs(image_complex[:, :nx // 2]) ** 2

		f_bin = s_new.astype(int)
		frac = s_new - f_bin

		for idx in np.ndindex(f_bin.shape):
			f = int(f_bin[idx])
			v = intensity[idx]
			if 0 <= f < npts:
				ret[f] += v * (1.0 - frac[idx])
				norm[f] += (1.0 - frac[idx])
			if f + 1 < npts:
				ret[f + 1] += v * frac[idx]
				norm[f + 1] += frac[idx]

		norm[norm == 0] = 1.0
		return ret / norm

	def compute_2d_real(self, ny, ctf_type, struct_factor=None):
		"""Compute a 2D real-space CTF.

		Currently unimplemented in the original C++ code (empty stub).
		Returns None as a placeholder.
		"""
		return None

	def compute_2d_complex(self, ny, ctf_type, struct_factor=None):
		"""Compute a 2D complex-frequency-space CTF multiplier.

		ny - image dimension in Y (X is ny+2 for even, ny+1 for odd)
		ctf_type - one of the CtfType constants
		struct_factor - optional numpy array for structure factor weighting

		Returns a complex64 numpy array of shape (ny, nx//2) ready to
		multiply with Fourier-domain image data element-wise.
		"""
		nx = ny + 2 if ny % 2 == 0 else ny + 1

		lam = self.wavelength()
		g1 = math.pi / 2.0 * self.cs * 1e7 * lam ** 3
		g2_coeff = math.pi * lam * 1e4
		acac = math.pi / 2.0 - self.get_phase()

		y_arr = np.arange(-ny // 2, ny // 2)
		x_arr = np.arange(nx // 2)
		X, Y = np.meshgrid(x_arr, y_arr)
		ds = 1.0 / (self.apix * ny)
		R = np.hypot(X, Y) * ds
		R_sq = R * R
		A = np.arctan2(Y, X)

		if self.dfdiff == 0:
			df_eff = self.defocus
		else:
			df_eff = self.defocus + self.dfdiff / 2.0 * np.cos(
				2.0 * A - 2.0 * math.pi / 180.0 * self.dfang
			)
		gam = -g1 * R_sq ** 2 + g2_coeff * df_eff * R_sq
		bf_env = np.exp(-(self.bfactor / 4.0) * R_sq)

		result = np.ones((ny, nx // 2), dtype=np.complex64)

		if ctf_type == CtfType.CTF_BACKGROUND:
			result.real = self._interpolate_background_2d(R)

		elif ctf_type == CtfType.CTF_AMP:
			v = np.cos(gam - acac) * bf_env
			result.real = v

		elif ctf_type == CtfType.CTF_ABS:
			v = np.cos(gam - acac) * bf_env
			result.real = np.abs(v)

		elif ctf_type == CtfType.CTF_INTEN:
			v = np.cos(gam - acac) * bf_env
			result.real = v * v

		elif ctf_type == CtfType.CTF_SIGN:
			v = np.cos(gam - acac)
			result.real = np.where(v < 0, -1.0, 1.0)

		elif ctf_type == CtfType.CTF_POWEVAL:
			mf = np.ones_like(R)
			mask_low = R < 0.04
			mask_trans = (R >= 0.04) & (R < 0.05)
			mf[mask_low] = 0.0
			mf[mask_trans] = 1.0 - np.exp(-((R[mask_trans] - 0.04) * 300.0) ** 2)
			v = np.cos(gam - acac)
			part = np.where(v > 0.9, np.exp(-(50.0 / 4.0 * R_sq)), 0.0)
			result.real = mf * part * part

		elif ctf_type == CtfType.CTF_FITREF:
			mf = np.ones_like(R)
			mask_low = R < 0.04
			mf[mask_low] = 0.0
			mask_trans = (R >= 0.04) & (R < 0.05)
			mf[mask_trans] = 1.0 - np.exp(-((R[mask_trans] - 0.04) * 300.0) ** 2)
			v = np.cos(gam - acac) * bf_env
			result.real = mf * v * v

		elif ctf_type == CtfType.CTF_SNR or ctf_type == CtfType.CTF_SNR_SMOOTH:
			if not self.snr:
				raise CtfTypeError("CTF_SNR requires SNR data")
			snr_vals = np.array(self.snr, dtype=np.float32)
			f = R / self.dsbg
			j = np.floor(f).astype(int)
			frac = f - j
			j_clipped = np.clip(j, 0, len(snr_vals) - 2)
			interp = snr_vals[j_clipped] * (1.0 - frac) + snr_vals[j_clipped + 1] * frac
			result.real = interp
			result[ny // 2, 0] = 0.0

		elif ctf_type == CtfType.CTF_WIENER_FILTER:
			if not self.snr:
				raise CtfTypeError("CTF_WIENER_FILTER requires SNR data")
			snr_vals = np.array(self.snr, dtype=np.float32)
			f = R / self.dsbg
			j = np.floor(f).astype(int)
			frac = f - j
			j_clipped = np.clip(j, 0, len(snr_vals) - 2)
			snr_interp = snr_vals[j_clipped] * (1.0 - frac) + snr_vals[j_clipped + 1] * frac
			snr_interp[snr_interp < 0] = 0.0
			result.real = snr_interp / (snr_interp + 1.0)
			result[ny // 2, 0] = 0.0

		elif ctf_type == CtfType.CTF_TOTAL:
			v = np.cos(gam - acac) * bf_env
			noise_vals = self._interpolate_background_2d(R)
			result.real = v * v + noise_vals

		elif ctf_type == CtfType.CTF_ALIFILT:
			v = np.cos(gam - acac) ** 2 * bf_env
			result.real = v

		return result.astype(np.complex64)

	def __repr__(self):
		return (f"EMAN2Ctf(defocus={self.defocus:.1f}, dfdiff={self.dfdiff:.1f}, "
		        f"dfang={self.dfang:.0f}, voltage={self.voltage:.1f}kV, cs={self.cs:.2f}mm, "
		        f"ampcont={self.ampcont:.1f}%, apix={self.apix:.3f})")
