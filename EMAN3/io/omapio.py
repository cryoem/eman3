#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ OmapIO)
# Copyright (c) 2000-2026 Baylor College of Medicine
#
# This software is issued under a joint BSD/GPL license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""OMAP/DNS6/BRIX map format reader/writer for EMAN.

OMAP (DnS6 MAP) is composed of records that are all 512 bytes long.
The first record may contain human-readable ASCII text (cell constants, etc.)
which is skipped if present. The subsequent record contains a 256-short integer
header with volume metadata. Data follows as 8x8x8 cubies packed into 512-byte
records. Each density sample is one unsigned byte.

Extensions: .omap, .dns6, .brix

Data layout: "x fast, y medium, z slow" (row-major). Cubie coordinates within
a record are ordered as z (slowest), then y, then x (fastest):
    record[8*8*n + 8*m + l] where n=local_z, m=local_y, l=local_x

Scaling: stored values are packed uint8 with linear transformation:
    real_value = (stored_value - iplus) / (iprod / scale2)
    Optionally divided by sigma if sigma > 0.

References: http://www.uoxray.uoregon.edu/tnt/manual/node104.html
"""

import os
import sys
import numpy as np


class FileFormatError(Exception):
	"""Exception for problems with file formats"""
	pass


class FileIOError(Exception):
	"""Exception for problems with file IO"""
	pass


class InvalidDimensions(Exception):
	"""Indicates a mismatch between data size and header indicated size"""
	pass


# 512 bytes = 256 x short (int16). Header layout from C++ OmapHeader struct.
# Always big-endian on disk regardless of host architecture.
OMAP_HEADER_DTYPE = np.dtype([
	("xstart",   ">i2"),     # origin x (grid index)
	("ystart",   ">i2"),     # origin y (grid index)
	("zstart",   ">i2"),     # origin z (grid index)
	("nx",       ">i2"),     # extent along x (grid steps)
	("ny",       ">i2"),     # extent along y (grid steps)
	("nz",       ">i2"),     # extent along z (grid steps)
	("sampling_x", ">i2"),   # total grid intervals across unit cell X
	("sampling_y", ">i2"),   # total grid intervals across unit cell Y
	("sampling_z", ">i2"),   # total grid intervals across unit cell Z
	("header10", ">i2"),     # scale * A cell edge
	("header11", ">i2"),     # scale * B cell edge
	("header12", ">i2"),     # scale * C cell edge
	("alpha",    ">i2"),     # alpha angle (degrees * scale)
	("beta",     ">i2"),     # beta angle  (degrees * scale)
	("gamma",    ">i2"),     # gamma angle (degrees * scale)
	("iprod",    ">i2"),     # data scaling: prod = iprod / scale2
	("iplus",    ">i2"),     # data offset: real = (stored - iplus) / prod
	("scale",    ">i2"),     # cell constant scaling factor (default 100)
	("scale2",   ">i2"),     # constant 100
	("imin",     ">i2"),     # minimum stored density value
	("imax",     ">i2"),     # maximum stored density value
	("isigma",   ">i2"),     # sigma (0 = no normalization)
	("imean",    ">i2"),     # mean density
	("unused",   ">i2", (233,)),  # unused space, always zero
])


def _is_ascii_record(record_bytes):
	"""Check if a 512-byte record is printable ASCII (preamble detection).

	Matches C++ logic: scan for non-printable chars (outside 32-126)
	or null terminator. If all bytes are printable, it's an ASCII preamble.
	"""
	for byte in record_bytes:
		if byte == 0:
			return False
		if byte < 32 or byte > 126:
			return False
	return True


def _swap_byte_pairs(record_array):
	"""Swap adjacent byte pairs to emulate big-endian uint16 on little-endian hosts.

	The OMAP format stores each 512-byte record as 256 big-endian uint16 words.
	On a little-endian host, we need to swap each pair of bytes so the individual
	uint8 samples end up in the correct order. Matches C++ behavior:
	    for (int ii=0; ii < 511; ii+=2) { swap(record[ii], record[ii+1]); }
	"""
	swapped = np.empty_like(record_array)
	swapped[0::2] = record_array[1::2]
	swapped[1::2] = record_array[0::2]
	return swapped


class OmapIO:
	"""OMAP/DNS6/BRIX map format reader/writer.

	Reads and writes a single 3D volume stored as packed uint8 cubies with
	linear intensity scaling. Supports .omap, .dns6, and .brix extensions.

	Modes: "r" (read), "rw" (write).
	Write creates files without ASCII preamble for simplicity.

	Usage example::

		# Write
		with OmapIO("volume.dns6", "rw") as omap:
			omap.write_header({"nx": 48, "ny": 32, "nz": 24}, 0)
			omap.write_data(volume_float32, 0)

		# Read
		with OmapIO("volume.dns6", "r") as omap:
			meta = omap.read_header(0)
			data = omap.read_data(0)   # shape (nz, ny, nx), float32

	Attributes (read-only after init):
		filename - path to the map file.
		nx, ny, nz - volume dimensions from header.
	"""

	# Class capability flags
	SUPPORT_STACK = False
	SUPPORT_3D = True
	SUPPORT_3D_STACK = False
	SUPPORTED_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode

		# Validate extension
		known_exts = (".omap", ".dns6", ".brix")
		base, ext = os.path.splitext(self.filename)
		if ext.lower() not in known_exts:
			raise FileFormatError(f"Unknown extension '{ext}' for OMAP format. "
			                      f"Expected one of {known_exts}")

		self._initialized = False
		self.nx = None
		self.ny = None
		self.nz = None
		self._header = None

		if mode == "r":
			try:
				self._file = open(self.filename, "rb")
			except OSError as e:
				raise FileIOError(f"Cannot open file '{self.filename}': {e}")
			self._parse_header()
		else:
			try:
				self._file = open(self.filename, "wb")
			except OSError as e:
				raise FileIOError(f"Cannot create file '{self.filename}': {e}")
			# Header is written by write_header

		self._initialized = True

	def __del__(self):
		if hasattr(self, '_file') and self._file is not None:
			try:
				self._file.close()
			except Exception:
				pass

	def _parse_header(self):
		"""Read and parse the OmapHeader.

		Skips optional ASCII preamble record if present. Header is always
		big-endian as per C++ convention. Returns parsed header struct.
		"""
		self._file.seek(0)

		# Read first 512-byte record to check for ASCII preamble
		record = self._file.read(512)
		if len(record) < 512:
			raise FileIOError("Incomplete OMAP header read")

		# C++ logic: scan for non-printable chars or null terminator.
		# If all printable, this is an ASCII preamble - skip it and read next.
		if _is_ascii_record(record):
			record = self._file.read(512)
			if len(record) < 512:
				raise FileIOError("Incomplete OMAP header read after ASCII preamble")

		self._header = np.frombuffer(record, dtype=OMAP_HEADER_DTYPE)[0]

		# Validate using C++ is_valid logic
		scale2_val = int(self._header["scale2"])
		if scale2_val != 100:
			raise FileFormatError(f"Invalid OMAP header: scale2={scale2_val} (expected 100)")

		self.nx = int(self._header["nx"])
		self.ny = int(self._header["ny"])
		self.nz = int(self._header["nz"])

		if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
			raise FileFormatError(f"Invalid OMAP dimensions: {self.nx}x{self.ny}x{self.nz}")
		if self.nx > 10000 or self.ny > 10000 or self.nz > 10000:
			raise FileFormatError(f"OMAP dimensions too large: {self.nx}x{self.ny}x{self.nz}")

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. OMAP is uint8 only."""
		return [8]

	def __len__(self):
		"""Return 1 — this format stores a single image."""
		return 1

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check whether a file has a valid OMAP/DNS6/BRIX header.

		Reads the first 512-byte record(s) and validates: scale2==100,
		dimensions in range (0, 10000], etc. Matches C++ OmapIO::is_valid.
		If chunk is provided, checks the chunk instead of reopening the file.
		"""
		try:
			if chunk is not None:
				# Use chunk data directly
				record = chunk[:512]
				if len(record) < 512:
					return False
				# Check for ASCII preamble (skip if present)
				if _is_ascii_record(record):
					record = chunk[512:1024]
					if len(record) < 512:
						return False
			else:
				with open(os.path.expanduser(filepath), "rb") as f:
					record = f.read(512)
					if len(record) < 512:
						return False
					# Check for ASCII preamble (skip if present)
					if _is_ascii_record(record):
						record = f.read(512)
						if len(record) < 512:
							return False

			hed = np.frombuffer(record, dtype=OMAP_HEADER_DTYPE)[0]

			# C++ is_valid checks scale2==100 and dimensions in (0, 10000]
			scale2_val = int(hed["scale2"])
			if scale2_val != 100:
				return False

			nx = int(hed["nx"])
			ny = int(hed["ny"])
			nz = int(hed["nz"])
			if nx <= 0 or ny <= 0 or nz <= 0:
				return False
			if nx > 10000 or ny > 10000 or nz > 10000:
				return False

			return True

		except Exception:
			return False

	def read_header(self, index=0):
		"""Read the OMAP header and return it as a dict.

		Returns:
			dict with keys: nx, ny, nz, bitdepth (always 8), min, max,
			mean, sigma, apix_x/y/z, cell constants, etc.
		"""
		if index != 0:
			raise FileFormatError("OMAP is single-volume format; index must be 0")

		h = self._header

		# Compute scaling factors (matching C++ read_header)
		scale2_val = int(h["scale2"]) if int(h["scale2"]) else 100
		scale_val = int(h["scale"]) if int(h["scale"]) else 100
		prod = float(h["iprod"]) / scale2_val
		plus = float(h["iplus"])

		# cell_edge = header field / scale, apix = cell_edge / sampling
		cell_a = int(h["header10"]) // max(1, scale_val) if scale_val else 0
		cell_b = int(h["header11"]) // max(1, scale_val) if scale_val else 0
		cell_c = int(h["header12"]) // max(1, scale_val) if scale_val else 0
		sm_x = max(1, int(h["sampling_x"]))
		sm_y = max(1, int(h["sampling_y"]))
		sm_z = max(1, int(h["sampling_z"]))

		result = {
			"nx": int(h["nx"]),
			"ny": int(h["ny"]),
			"nz": int(h["nz"]),
			"bitdepth": 8,
			"minimum": (float(h["imin"]) - plus) / prod if prod != 0 else 0.0,
			"maximum": (float(h["imax"]) - plus) / prod if prod != 0 else 0.0,
			"mean": (float(h["imean"]) - plus) / prod if prod != 0 else 0.0,
			"sigma": (float(h["isigma"]) - plus) / prod if prod != 0 else 0.0,
			"sampling_x": int(h["sampling_x"]),
			"sampling_y": int(h["sampling_y"]),
			"sampling_z": int(h["sampling_z"]),
			"apix_x": float(cell_a / sm_x),
			"apix_y": float(cell_b / sm_y),
			"apix_z": float(cell_c / sm_z),
			"OMAP.xstart": int(h["xstart"]),
			"OMAP.ystart": int(h["ystart"]),
			"OMAP.zstart": int(h["zstart"]),
			"OMAP.cellA": cell_a,
			"OMAP.cellB": cell_b,
			"OMAP.cellC": cell_c,
			"alpha": int(h["alpha"]) // max(1, scale_val) if scale_val else 0,
			"beta": int(h["beta"]) // max(1, scale_val) if scale_val else 0,
			"gamma": int(h["gamma"]) // max(1, scale_val) if scale_val else 0,
			"OMAP.iprod": int(h["iprod"]),
			"OMAP.iplus": int(h["iplus"]),
			"OMAP.scale": int(h["scale"]) if int(h["scale"]) else 100,
			"OMAP.scale2": int(h["scale2"]) if int(h["scale2"]) else 100,
		}

		return result

	def read_data(self, index=0):
		"""Read the OMAP volume data as float32.

		Data is stored as packed uint8 cubies with linear scaling:
			value = (stored_byte - iplus) / (iprod / scale2)
		If sigma > 0, values are additionally divided by sigma (matching C++).

		Returns a numpy array of shape (nz, ny, nx) with dtype float32.
		X-fast ordering: x is fastest axis, z is slowest.

		Parameters:
			index : int  Must be 0 (single volume format).

		Returns:
			numpy.ndarray[float32] shaped (nz, ny, nx).
		"""
		if index != 0:
			raise FileFormatError("OMAP is single-volume format; index must be 0")

		h = self._header
		nx, ny, nz = int(h["nx"]), int(h["ny"]), int(h["nz"])

		# Number of cubies in each direction (8 voxels per cubie)
		inx = (nx + 7) // 8
		iny = (ny + 7) // 8
		inz = (nz + 7) // 8

		# Remainder for last partial cubie in each direction
		xtra_x = nx % 8
		xtra_y = ny % 8
		xtra_z = nz % 8

		# Scaling parameters
		scale2_val = int(h["scale2"]) if int(h["scale2"]) else 100
		prod = float(h["iprod"]) / scale2_val
		plus = float(h["iplus"])
		sigma_raw = int(h["isigma"])

		data = np.zeros((nz, ny, nx), dtype=np.float32)

		# On little-endian hosts, swap adjacent byte pairs in each record.
		do_swap = (sys.byteorder != 'big')

		for k in range(inz):
			cubie_z_size = xtra_z if (xtra_z > 0 and k == inz - 1) else 8
			for j in range(iny):
				cubie_y_size = xtra_y if (xtra_y > 0 and j == iny - 1) else 8
				for i in range(inx):
					cubie_x_size = xtra_x if (xtra_x > 0 and i == inx - 1) else 8

					record = self._file.read(512)
					if len(record) < 512:
						raise FileIOError("Incomplete OMAP data read")

					# Extract individual bytes from the record
					cubie_bytes = np.frombuffer(record, dtype=np.uint8)

					# Swap adjacent byte pairs on little-endian (matches C++)
					if do_swap:
						cubie_bytes = _swap_byte_pairs(cubie_bytes)

					# Reshape to full 8x8x8 cubie (z, y, x ordering per spec)
					cubie_3d = cubie_bytes[:512].reshape((8, 8, 8))

					# Take only the valid portion of this cubie
					valid = cubie_3d[:cubie_z_size, :cubie_y_size, :cubie_x_size].astype(
						np.float32)

					# Apply linear scaling transformation
					if prod != 0:
						valid = (valid - plus) / prod
					else:
						valid[:] = 0.0

					if sigma_raw > 0:
						valid /= float(sigma_raw)

					# Copy into the output volume buffer
					z_start = k * 8
					y_start = j * 8
					x_start = i * 8
					data[z_start:z_start + cubie_z_size,
						 y_start:y_start + cubie_y_size,
						 x_start:x_start + cubie_x_size] = valid

		return data

	def write_header(self, d, index=0):
		"""Write the OMAP header.

		Parameters:
		d : dict with keys nx, ny, nz (required). Optional: apix_x/y/z
		(in angstroms/pixel), sampling_x/y/z (grid intervals per cell,
		preferred), alpha/beta/gamma (in degrees), minimum/maximum.
			If neither apix nor sampling is given, defaults to 1 pixel/unit.
		 Cell edges default to nx, ny, nz. Angles default to 90 deg.
		index : int  Must be 0 (single volume format).

		Returns:
			0 on success.
		"""
		if self.mode != "rw":
			raise FileIOError("OMAP file opened in read-only mode")

		if index != 0:
			raise FileFormatError("OMAP is single-volume format; index must be 0")

		nx = int(d["nx"])
		ny = int(d["ny"])
		nz = int(d["nz"])

		if nx <= 0 or ny <= 0 or nz <= 0:
			raise FileFormatError(f"Invalid OMAP dimensions: {nx}x{ny}x{nz}")
		if nx > 10000 or ny > 10000 or nz > 10000:
			raise FileFormatError(f"OMAP dimensions too large: {nx}x{ny}x{nz}")

		self.nx = nx
		self.ny = ny
		self.nz = nz

		# Scale factor for cell constants and angles (convention: 100)
		scale_val = int(d.get("OMAP.scale", 100)) if d.get("OMAP.scale") else 100
		if scale_val == 0:
			scale_val = 100

		# Cell edges, sampling and derived apix:
		# Entry 7-9: sampling = total grid intervals across unit cell
		# Entry 10-12: scale * cell_edge
		# apix (angstroms/pixel) = cell_edge / sampling
		cell_a = d.get("OMAP.cellA", nx)
		cell_b = d.get("OMAP.cellB", ny)
		cell_c = d.get("OMAP.cellC", nz)
		apix_x_val = d.get("apix_x", None)
		apix_y_val = d.get("apix_y", None)
		apiz_z_val = d.get("apiz", None)
		sampling_x = int(d.get("sampling_x", max(1, round(cell_a / (apix_x_val or 1.0)))))
		sampling_y = int(d.get("sampling_y", max(1, round(cell_b / (apix_y_val or 1.0)))))
		sampling_z = int(d.get("sampling_z", max(1, round(cell_c / (apiz_z_val or 1.0)))))

		# Angles in degrees (stored as degrees * 100, default 90 deg = 9000)
		alpha_deg = d.get("alpha", 90)
		beta_deg = d.get("beta", 90)
		gamma_deg = d.get("gamma", 90)

		# Compute iprod/iplus from data min/max for proper scaling.
		# Formula: real = (stored - iplus) / (iprod / scale2)
		# Inverse: stored = real * (iprod/scale2) + iplus
		# Target: stored_min ~ 0, stored_max ~ 255
		data_min = d.get("minimum", 0.0)
		data_max = d.get("maximum", 1.0)

		if data_min is None:
			data_min = 0.0
		if data_max is None:
			data_max = 1.0

		data_range = abs(data_max - data_min)
		total_span = abs(data_min) + data_range if data_min != 0 else data_range

		if total_span > 0:
			prod = 254.0 / total_span   # Use 254 not 255 to leave headroom
		else:
			prod = 1.0

		iprod = max(1, int(round(prod * 100)))
		iplus = int(round(-data_min * prod))

		hed = np.zeros((1,), dtype=OMAP_HEADER_DTYPE)
		hed[0]["nx"] = nx
		hed[0]["ny"] = ny
		hed[0]["nz"] = nz
		hed[0]["sampling_x"] = sampling_x
		hed[0]["sampling_y"] = sampling_y
		hed[0]["sampling_z"] = sampling_z
		hed[0]["header10"] = int(cell_a * scale_val)
		hed[0]["header11"] = int(cell_b * scale_val)
		hed[0]["header12"] = int(cell_c * scale_val)
		hed[0]["alpha"] = int(round(alpha_deg * 100))
		hed[0]["beta"] = int(round(beta_deg * 100))
		hed[0]["gamma"] = int(round(gamma_deg * 100))
		hed[0]["iprod"] = iprod
		hed[0]["iplus"] = iplus
		hed[0]["scale"] = scale_val
		hed[0]["scale2"] = 100
		hed[0]["isigma"] = 0  # No sigma normalization
		hed[0]["unused"][:] = 0

		# Compute stored min/max from the scaling parameters
		stored_min = int(round(data_min * prod + iplus))
		stored_max = int(round(data_max * prod + iplus))
		stored_min = max(0, min(255, stored_min))
		stored_max = max(0, min(255, stored_max))
		hed[0]["imin"] = stored_min
		hed[0]["imax"] = stored_max
		hed[0]["imean"] = int(round(stored_min + (stored_max - stored_min) / 2))

		self._header = hed[0]

		# Write header record as big-endian uint16 (no ASCII preamble)
		self._file.seek(0)
		self._file.write(hed.tobytes())

		return 0

	def write_data(self, data, index=0):
		"""Write volume data to the OMAP file.

		The input float32 data is scaled to uint8 using the iprod/iplus
		parameters from the header written by write_header. Values are
		clipped to [0, 255] and packed into 8x8x8 cubie records.

		On little-endian hosts, each record is constructed as big-endian
		uint16 words where even-indexed pixels go into the low byte and
		odd-indexed pixels go into the high byte (so that byte-swap on read
		restores correct ordering).

		Parameters:
			data : numpy.ndarray of shape (nz, ny, nx), float32.
			index : int  Must be 0 (single volume format).
		"""
		if self.mode != "rw":
			raise FileIOError("OMAP file opened in read-only mode")

		if index != 0:
			raise FileFormatError("OMAP is single-volume format; index must be 0")

		if self._header is None:
			raise FileIOError("write_header must be called before write_data")

		nx, ny, nz = self.nx, self.ny, self.nz

		# Ensure data has correct shape
		if len(data.shape) == 3:
			shape = data.shape
		elif len(data.shape) == 2:
			shape = (1, data.shape[0], data.shape[1])
		else:
			raise FileFormatError(f"Invalid data shape: {data.shape}")

		if shape != (nz, ny, nx):
			raise InvalidDimensions(
				f"Data shape {shape} doesn't match header dimensions ({nz},{ny},{nx})")

		data = np.asarray(data, dtype=np.float32)

		# Scaling parameters from header
		scale2_val = int(self._header["scale2"]) if int(self._header["scale2"]) else 100
		prod = float(self._header["iprod"]) / scale2_val
		plus = float(self._header["iplus"])

		# Scale to uint8: stored = round(real * prod + plus)
		if prod != 0:
			stored = np.round(data * prod + plus).clip(0, 255).astype(np.uint8)
		else:
			stored = np.zeros((nz, ny, nx), dtype=np.uint8)

		# Pad to full cubie dimensions (next multiple of 8)
		pad_nz = ((nz + 7) // 8) * 8
		pad_ny = ((ny + 7) // 8) * 8
		pad_nx = ((nx + 7) // 8) * 8

		padded = np.zeros((pad_nz, pad_ny, pad_nx), dtype=np.uint8)
		padded[:nz, :ny, :nx] = stored

		# Number of cubie records
		inx = pad_nx // 8
		iny = pad_ny // 8
		inz = pad_nz // 8

		for k in range(inz):
			for j in range(iny):
				for i in range(inx):
					z_start = k * 8
					y_start = j * 8
					x_start = i * 8

					# Extract 8x8x8 cubie as flat 512 bytes (z-slow, y-medium, x-fast)
					cubie = padded[z_start:z_start + 8,
					               y_start:y_start + 8,
					               x_start:x_start + 8].ravel()

					# Pack into big-endian uint16 words. On-disk format is 256 BE uint16.
					# Each word packs two adjacent pixels: low byte = even index,
					# high byte = odd index. After fread + byte-swap on LE hosts,
					# this restores correct pixel ordering (verified empirically).
					words = np.zeros(256, dtype='>u2')
					for p in range(512):
						word_pos = p >> 1
						val = int(cubie[p]) & 0xFF
						if p % 2 == 0:
							words[word_pos] |= val          # low byte of BE word
						else:
							words[word_pos] |= (val << 8)   # high byte of BE word

					self._file.write(words.tobytes())

	def flush(self):
		"""Flush the file buffer."""
		if self._file is not None:
			self._file.flush()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None:
			try:
				self._file.close()
			except Exception:
				pass
		self._file = None
		return False
