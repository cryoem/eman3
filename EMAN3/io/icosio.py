#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ IcosIO)
# Copyright (c) 2000-2006 Baylor College of Medicine
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
# Foundation, Inc. 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""ICOS image format reader/writer for EMAN.

ICOS (ICONS) stores floating-point image data with a simple binary header
and padded row data. Each row is bracketed by sentinel integers:

	Row layout: [sentinel_int] [nx floats] [sentinel_int]
	where sentinel_int = nx * sizeof(float) = nx * 4

The header is ~112 bytes containing stamps, title, dimensions, and min/max stats.
Single image per file — no stacks. Supports 2D or 3D data.

References: Internal EMAN format (not widely documented externally).
"""

import os
import struct
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

class InvalidIndex(Exception):
	"""Indicates a request for an unsupported or missing image index"""
	pass


# ICOS header constants
ICOS_STAMP = 72
ICOS_STAMP1 = 72
ICOS_STAMP2 = 20
ICOS_STAMP3 = 20

# Header layout:
#   stamp      int   (=72)       offset 0
#   title      char[72]          offset 4
#   stamp1     int   (=72)       offset 76
#   stamp2     int   (=20)       offset 80
#   nx         int              offset 84
#   ny         int              offset 88
#   nz         int              offset 92
#   min        float            offset 96
#   max        float            offset 100
#   stamp3     int   (=20)       offset 104
ICOS_HEADER_FORMAT = "<i72siiiiiffi"
ICOS_HEADER_SIZE = struct.calcsize(ICOS_HEADER_FORMAT)


class IcosIO:
	"""Read and write ICOS format image files.

	ICOS stores float32 image data (2D or 3D) with a simple binary header
	and row-padded data. Single image per file - no stacks.

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with IcosIO("myfile.ico", "r") as icos:
			meta = icos.read_header(0)
			data = icos.read_data(0)   # numpy array (nz, ny, nx)

		with IcosIO("out.ico", "rw") as icos:
			icos.write_header({"nx":128, "ny":128, "nz":1})
			icos.write_data(np.random.rand(1, 128, 128).astype(np.float32), 0)

	Attributes (read-only after init):
		filename   - path to the ICOS file.
		nx, ny, nz - dimensions from the header.
		big_endian - endianness of the on-disk data.
	"""

	# Class capability flags
	SUPPORT_STACK = False
	SUPPORT_3D = True
	SUPPORT_3D_STACK = False
	SUPPORT_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode
		self._file = None
		self._is_new_file = not os.path.exists(self.filename)
		self._initialized = False

		self.nx = None
		self.ny = None
		self.nz = None
		self.ic_min = 0.0
		self.ic_max = 0.0
		self.big_endian = False

		if self._is_new_file:
			fmode = "wb+"
			self._initialized = True
		elif mode == "r":
			fmode = "rb"
			self._initialized = True
		else:
			fmode = "rb+"
			self._initialized = True

		try:
			self._file = open(self.filename, fmode)
		except OSError as e:
			raise FileIOError(f"Cannot open file '{self.filename}': {e}")

		if self._initialized and not self._is_new_file:
			self._read_header()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None:
			self._file.close()
			self._file = None
		return False

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. ICOS is float32 only (bitdepth=0)."""
		return [0]

	def __len__(self):
		"""Return 1 — this format stores a single image."""
		return 1

	def is_valid(filepath, chunk):
		"""Check whether a file is potentially a valid ICOS image.

		Checks the stamp fields at known offsets in the header.
		Tries both endianness.
		"""
		try:
			if len(chunk) < ICOS_HEADER_SIZE:
				return False

			for endian_prefix in ("<", ">"):
				fmt = endian_prefix + "i72siiiiiffi"
				fields = struct.unpack_from(fmt, chunk)
				stamp, title, stamp1, stamp2, nx, ny, nz, fmin, fmax, stamp3 = fields
				if stamp == ICOS_STAMP and stamp1 == ICOS_STAMP1 and stamp2 == ICOS_STAMP2 and stamp3 == ICOS_STAMP3:
					if nx > 0 and ny > 0:
						return True

			return False
		except Exception:
			return False

	def _read_header(self):
		"""Read and parse the ICOS header. Sets nx, ny, nz, big_endian, etc."""
		self._file.seek(0)
		raw = self._file.read(ICOS_HEADER_SIZE)
		if len(raw) < ICOS_HEADER_SIZE:
			raise FileFormatError(f"Incomplete ICOS header read (got {len(raw)} of {ICOS_HEADER_SIZE} bytes)")

		# Try both endianness to detect and parse
		for endian in ("<", ">"):
			fmt = endian + "i72siiiiiffi"
			fields = struct.unpack_from(fmt, raw)
			stamp, title, stamp1, stamp2, nx, ny, nz, fmin, fmax, stamp3 = fields
			if stamp == ICOS_STAMP and stamp1 == ICOS_STAMP1:
				self.big_endian = (endian == ">")
				self.nx = nx
				self.ny = ny
				self.nz = nz
				self.ic_min = fmin
				self.ic_max = fmax
				return

		raise FileFormatError("Invalid ICOS file stamps in header")

	def read_header(self, index=0):
		"""Read the ICOS header and return it as a dict.

		ICOS is a single-image format; index must be 0 (or -1, treated as 0).

		Returns:
			dict with keys: nx, ny, nz, bitdepth (always 0), minimum, maximum.
		"""
		if index != 0 and index != -1:
			raise InvalidIndex("ICOS is a single-image format; index must be 0")

		if not self._initialized:
			self._read_header()

		return {
			"nx": self.nx,
			"ny": self.ny,
			"nz": self.nz,
			"bitdepth": 0,
			"minimum": self.ic_min,
			"maximum": self.ic_max,
		}

	def read_data(self, index=0):
		"""Read the float32 image data.

		Returns a numpy array of shape (nz, ny, nx) with dtype float32.
		Each row in the file is padded with sentinel integers at start and end,
		which are stripped during read.

		Parameters:
			index : int  Image number (must be 0 for ICOS).

		Returns:
			numpy.ndarray[float32] shaped (nz, ny, nx).
		"""
		if index != 0 and index != -1:
			raise InvalidIndex("ICOS is a single-image format; index must be 0")

		self._file.seek(ICOS_HEADER_SIZE)

		dt = '<f4'  # ICOS always little-endian
		self.big_endian = False

		nrows = self.ny * self.nz
		data = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)

		for k in range(nrows):
			row_raw = self._file.read(8 + self.nx * 4)
			if len(row_raw) < (8 + self.nx * 4):
				raise FileIOError(f"Incomplete ICOS data read at row {k}")

			# Parse sentinel + data + sentinel as floats, strip sentinels
			row_floats = np.frombuffer(row_raw, dtype=dt)
			row_data = row_floats[1:-1].copy()

			# Place into 3D array
			z = k // self.ny
			y = k % self.ny
			data[z, y, :] = row_data

		return data

	def write_header(self, meta, index=-1):
		"""Write an ICOS header.

		meta must contain "nx", "ny", and "nz".
		Optional keys: "minimum", "maximum" (computed from data if not provided).
		Index must be 0 (single-image format). Returns 0.
		"""
		if self.mode == "r":
			raise FileIOError("ICOS file opened read-only; writes forbidden")

		if index != -1 and index != 0:
			raise InvalidIndex("ICOS is a single-image format; index must be 0")

		self.nx = int(meta["nx"])
		self.ny = int(meta["ny"])
		self.nz = int(meta.get("nz", 1))

		bitdepth = int(meta.get("bitdepth", 0))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"ICOS only supports float32 (bitdepth=0), got {bitdepth}")

		self.ic_min = float(meta.get("minimum", 0.0))
		self.ic_max = float(meta.get("maximum", 0.0))

		self._file.seek(0)

		title = b"EMAN3 ICOS file" + b"\x00" * 62  # 72 bytes total
		header = struct.pack(
			"<i72siiiiiffi",
			ICOS_STAMP,
			title,
			ICOS_STAMP1,
			ICOS_STAMP2,
			self.nx,
			self.ny,
			self.nz,
			self.ic_min,
			self.ic_max,
			ICOS_STAMP3,
		)

		self._file.write(header)
		self._is_new_file = False
		self._initialized = True

		return 0

	def write_data(self, data, index=0):
		"""Write float32 image data.

		data should be a numpy array of shape (nz, ny, nx). Header must have
		been written first. Each row is written with sentinel integers at start
		and end as per the ICOS spec.
		"""
		if index != 0 and index != -1:
			raise InvalidIndex("ICOS is a single-image format; index must be 0")

		data = np.asarray(data, dtype=np.float32)

		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]

		if data.ndim == 2:
			data = data[np.newaxis, :]

		if data.ndim != 3:
			raise InvalidDimensions(f"Data must be 2D or 3D, got shape {data.shape}")

		nz, ny, nx = data.shape
		if nx != self.nx or ny != self.ny or nz != self.nz:
			raise InvalidDimensions(
				f"Data shape ({nz}, {ny}, {nx}) != header ({self.nz}, {self.ny}, {self.nx})")

		flipped = data.astype(np.float32)
		self.big_endian = False  # ICOS always little-endian
		sentinel_val = self.nx * 4  # bytes per row of float data
		self._file.seek(ICOS_HEADER_SIZE)

		for z in range(nz):
			for y in range(ny):
				row = flipped[z, y, :]
				row_bytes = np.empty(2 + nx, dtype='<f4')
				row_bytes[0] = sentinel_val
				row_bytes[1:-1] = row
				row_bytes[-1] = sentinel_val
				self._file.write(row_bytes.tobytes())
