#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ AmiraIO)
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

"""Amira Mesh format reader/writer for EMAN.

Amira uses a human-readable ASCII header followed by binary data. The header
defines dimensions (Lattice), BoundingBox, and data type. Binary data follows
an `@1` marker.

Supported extensions: .am, .amira, .mesh, .raw
Supported data types: float32, int16, uint8
Single 3D volume only - no stack support.

Note: Amira stores data with Y flipped relative to EMAN convention. The C++
code performs a Y-axis flip on both read and write to maintain consistency.
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


MAGIC = "# AmiraMesh"


def _swap_bytes_inplace(arr, itemsize):
	"""Swap bytes within each element of array to convert endianness.

	This matches the C++ behavior where individual bytes within each sample
	are reversed for byte-order conversion.
	"""
	if arr.dtype.byteorder == '<':
		arr = arr.byteswap(inplace=True)
	return arr


class AmiraIO:
	"""Amira Mesh format reader/writer.

	Reads and writes a single 3D volume with text header describing
	dimensions, bounding box, and data type. Binary data follows `@1` marker.

	Supports float32, int16 (short), and uint8 (byte) data types.
	Both big-endian and little-endian are supported.

	Modes: "r" (read), "rw" (write).

	Usage example::

		with AmiraIO("volume.am", "rw") as amira:
			amira.write_header({"nx": 48, "ny": 32, "nz": 24}, 0)
			amira.write_data(volume_float32, 0)

		with AmiraIO("volume.am", "r") as amira:
			meta = amira.read_header(0)
			data = amira.read_data(0)   # shape (nz, ny, nx), float32

	Attributes (read-only after init):
		filename - path to the Amira file.
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
		known_exts = (".am", ".amira", ".mesh", ".raw")
		base, ext = os.path.splitext(self.filename)
		if ext.lower() not in known_exts:
			raise FileFormatError(f"Unknown extension '{ext}' for Amira format. "
			                      f"Expected one of {known_exts}")

		self._initialized = False
		self.nx = None
		self.ny = None
		self.nz = None
		self._header_dict = {}
		self._datatype = None  # EM data type string: 'float', 'short', 'byte'
		self._is_big_endian = True
		self._pixel_size = 1.0
		self._xorigin = 0.0
		self._yorigin = 0.0
		self._zorigin = 0.0

		try:
			if mode == "r":
				self._file = open(self.filename, "rb")
				self._parse_header()
			else:
				self._file = open(self.filename, "wb")
		except OSError as e:
			raise FileIOError(f"Cannot open file '{self.filename}': {e}")

		self._initialized = True

	def __del__(self):
		if hasattr(self, '_file') and self._file is not None:
			try:
				self._file.close()
			except Exception:
				pass

	def _parse_header(self):
		"""Parse the ASCII header of an Amira file.

		Reads text lines until `@1` marker which indicates start of binary data.
		Matches C++ AmiraIO::read_header parsing logic.
		"""
		self._file.seek(0)

		# Read and validate magic line
		first_line = self._file.readline()
		if isinstance(first_line, bytes):
			first_line = first_line.decode('utf-8', errors='replace')

		if MAGIC not in first_line:
			raise FileFormatError("Invalid Amira Mesh file: missing magic string")

		# Detect endianness from first line
		if "BINARY-LITTLE-ENDIAN" in first_line:
			self._is_big_endian = False
		elif "BINARY" in first_line or "2.0" in first_line or "2.1" in first_line:
			self._is_big_endian = True

		nx = ny = nz = 0
		datatype = None
		pixel_size = 1.0
		xorigin = yorigin = zorigin = 0.0

		# Read header lines until @1 marker
		while True:
			line = self._file.readline()
			if isinstance(line, bytes):
				line = line.decode('utf-8', errors='replace')

			# Parse Lattice data type FIRST (may be on same line as @1)
			if 'Lattice {' in line:
				parts = line.split()
				try:
					if len(parts) >= 3:
						datatype = parts[2]
				except IndexError:
					pass

			# Check for standalone @1 data marker that ends header
			if line.startswith('@1'):
				break

			# Parse Lattice dimensions
			if 'define Lattice' in line:
				parts = line.split()
				try:
					idx = parts.index('Lattice') + 1
					nx, ny, nz = int(parts[idx]), int(parts[idx+1]), int(parts[idx+2])
				except (ValueError, IndexError):
					pass

			# Parse BoundingBox (Amira 2.0 - 6 values) or BoundingBoxXY (2.1 - 4 values)
			if 'BoundingBox' in line and 'Lattice' not in line:
				parts = line.split()
				try:
					# Find numbers after "BoundingBox" key
					nums = [float(x) for x in parts if x.replace('.','').replace('-','').isdigit() or
					        (x[1:].replace('.','').replace('-','').isdigit() and x.startswith('-'))]

					if len(nums) >= 4:
						xorigin = nums[0]
						pixel_size = (nums[1] - nums[0]) / max(1, nx - 1) if nx > 1 else 1.0
						yorigin = nums[2]
						if len(nums) >= 6:
							zorigin = nums[4]
				except (ValueError, IndexError):
					pass


		if nx <= 0 or ny <= 0 or nz <= 0:
			raise FileFormatError(f"Invalid Amira dimensions: {nx}x{ny}x{nz}")
		if datatype is None:
			raise FileFormatError("Cannot determine data type from Amira header")

		self.nx = nx
		self.ny = ny
		self.nz = nz
		self._datatype = datatype.lower()
		self._pixel_size = pixel_size
		self._xorigin = xorigin
		self._yorigin = yorigin
		self._zorigin = zorigin
		self._datatype = 'float'  # Amira always writes float32

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. Amira supports 8, 16, and 32-bit."""
		return [8, 16, 32]

	def __len__(self):
		"""Return 1 — this format stores a single image."""
		return 1

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check whether a file has a valid Amira Mesh header.

		Reads the first line and checks for the magic string `# AmiraMesh`.
		If chunk is provided, checks the chunk instead of reopening the file.
		"""
		try:
			if chunk is not None:
				text = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else chunk
				return MAGIC in text
			with open(os.path.expanduser(filepath), 'rb') as f:
				first_line = f.readline()
				if isinstance(first_line, bytes):
					first_line = first_line.decode('utf-8', errors='replace')
				return MAGIC in first_line
		except Exception:
			return False

	def read_header(self, index=0):
		"""Read the Amira header and return it as a dict.

		Returns:
			dict with keys: nx, ny, nz, bitdepth, datatype, apix_x/y/z,
			origin_x/y/z.
		"""
		if index != 0:
			raise FileFormatError("Amira is single-image format; index must be 0")

		# Map Amira data type to bit depth
		type_map = {'float': 0, 'short': 16, 'byte': 8}
		bitdepth = type_map.get(self._datatype.lower(), 32)

		result = {
			"nx": self.nx,
			"ny": self.ny,
			"nz": self.nz,
			"bitdepth": bitdepth,
			"apix_x": self._pixel_size,
			"apix_y": self._pixel_size,
			"apiz_z": self._pixel_size,
			"origin_x": self._xorigin,
			"origin_y": self._yorigin,
			"origin_z": self._zorigin,
		}

		return result

	def read_data(self, index=0):
		"""Read the Amira volume data as float32.

		Data is read in native format (float/short/byte), converted to float32,
		and Y-flipped to match EMAN convention.

		Returns a numpy array of shape (nz, ny, nx) with dtype float32.

		Parameters:
			index : int  Must be 0 (single image format).

		Returns:
			numpy.ndarray[float32] shaped (nz, ny, nx).
		"""
		if index != 0:
			raise FileFormatError("Amira is single-image format; index must be 0")

		dt = self._datatype.lower()
		total_size = self.nx * self.ny * self.nz

		if dt == "float":
			dtype_str = ">f4" if self._is_big_endian else "<f4"
			data = np.fromfile(self._file, dtype=dtype_str, count=total_size)
		elif dt == "short":
			dtype_str = ">i2" if self._is_big_endian else "<i2"
			raw = np.fromfile(self._file, dtype=dtype_str, count=total_size)
			data = raw.astype(np.float32)
		elif dt == "byte":
			raw = np.fromfile(self._file, dtype=np.uint8, count=total_size)
			data = raw.astype(np.float32)
		else:
			raise FileFormatError(f"Unsupported Amira data type: {self._datatype}")

		# Reshape to C++ layout: ny rows of nz*nx floats each -> (ny, nz, nx)
		data = data.reshape((self.ny, self.nz, self.nx))

		# Y-flip: reverse along first axis (ny), matching C++ write_data flip
		data = data[::-1].copy()

		# Transpose back to EMAN convention: (ny, nz, nx) -> (nz, ny, nx)
		data = np.transpose(data, (1, 0, 2))

		return data
	def write_header(self, d, index=0):
		"""Write the Amira header as ASCII text.

		Parameters:
			d : dict with keys nx, ny, nz (required). Optional: apix_x for
				pixel size (default 1.0), origin_x/y/z (default 0.0).
			index : int  Must be 0 (single image format).

		Returns:
			0 on success.
		"""
		if self.mode != "rw":
			raise FileIOError("Amira file opened in read-only mode")

		if index != 0:
			raise FileFormatError("Amira is single-image format; index must be 0")

		self.nx = int(d["nx"])
		self.ny = int(d["ny"])
		self.nz = int(d["nz"])

		pixel = float(d.get("apix_x", 1.0))
		xorigin = float(d.get("origin_x", 0.0))
		yorigin = float(d.get("origin_y", 0.0))
		zorigin = float(d.get("origin_z", 0.0))

		self._pixel_size = pixel
		self._xorigin = xorigin
		self._yorigin = yorigin
		self._zorigin = zorigin
		self._datatype = 'float'  # Amira always writes float32

		# Determine endianness string
		if sys.byteorder == 'big':
			header_line = "# AmiraMesh 3D BINARY 2.1"
		else:
			header_line = "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"

		self._is_big_endian = (sys.byteorder != 'little')

		# Write header lines
		try:
			self._file.write(header_line.encode('utf-8'))
			self._file.write(b'\n\n')

			self._file.write(
				f'define Lattice {self.nx} {self.ny} {self.nz}\n\n'.encode('utf-8'))

			self._file.write(b'Parameters {\n')
			self._file.write(
				f'\tContent "{self.nx}x{self.ny}x{self.nz} float, uniform coordinates",\n'.encode('utf-8'))
			self._file.write(b'\tCoordType "uniform",\n')

			bbx1 = xorigin + pixel * (self.nx - 1)
			byy1 = yorigin + pixel * (self.ny - 1)
			bzz1 = zorigin + pixel * (self.nz - 1)
			self._file.write(
				f'\tBoundingBox {xorigin:.2f} {bbx1:.2f} {yorigin:.2f} '
				f'{byy1:.2f} {zorigin:.2f} {bzz1:.2f}\n'.encode('utf-8'))
			self._file.write(b'}\n\n')

			self._file.write(b'Lattice { float ScalarField } @1\n\n')
			self._file.write(b'@1\n')
		except OSError as e:
			raise FileIOError(f"Cannot write Amira header: {e}")

		return 0

	def write_data(self, data, index=0):
		"""Write volume data to the Amira file.

		Input float32 data is Y-flipped and written in native byte order.

		Parameters:
			data : numpy.ndarray of shape (nz, ny, nx), float32.
			index : int  Must be 0 (single image format).
		"""
		if self.mode != "rw":
			raise FileIOError("Amira file opened in read-only mode")

		if index != 0:
			raise FileFormatError("Amira is single-image format; index must be 0")

		# Ensure correct shape
		if data.shape != (self.nz, self.ny, self.nx):
			raise InvalidDimensions(
				f"Data shape {data.shape} doesn't match header "
				f"({self.nz},{self.ny},{self.nx})")

		data = np.asarray(data, dtype=np.float32)

		# Transpose to (ny, nz, nx) for Amira storage layout
		# C++ fread reads ny rows of nx*nz floats each → reshape (ny, nz*nx)
		data = np.transpose(data, (1, 0, 2))

		# Y-flip: reverse along first axis (ny) before writing
		data = data[::-1].copy()

		# Flatten to write as contiguous bytes
		if self._is_big_endian and sys.byteorder == 'little':
			data.astype('>f4').tofile(self._file)
		elif not self._is_big_endian and sys.byteorder == 'big':
			data.astype('<f4').tofile(self._file)
		else:
			data.tofile(self._file)

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
