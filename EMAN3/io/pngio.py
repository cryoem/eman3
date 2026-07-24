#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ PngIO)
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

"""PNG image format reader/writer for EMAN.

PNG supports lossless compression of grayscale images at 8-bit or 16-bit depth.
Single image per file — no stacks.

Read support includes auto-detection of bit depth, color-to-grayscale conversion,
and Y-axis flip to match EMAN convention.

References: https://www.w3.org/TR/PNG/
"""

import os
import numpy as np
from PIL import Image


class FileFormatError(Exception):
	"""Exception for problems with file formats"""
	pass

class FileIOError(Exception):
	"""Exception for problems with file IO"""
	pass

class InvalidDimensions(Exception):
	"""Indicates a mismatch between data size and header indicated size"""
	pass


class PngIO:
	"""Read and write PNG format image files.

	PNG stores a single 2D grayscale image at 8-bit or 16-bit depth with
	lossless compression. On read, bit depth is auto-detected. On write,
	bit depth is controlled by "PNG.bitdepth" in metadata (default 16).

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with PngIO("out.png", "rw") as png:
			png.write_header({"nx": 256, "ny": 256, "PNG.bitdepth": 16})
			png.write_data(image_float, 0)

		with PngIO("in.png", "r") as png:
			meta = png.read_header()
			data = png.read_data(0)   # shape (ny, nx), float32

	Attributes (read-only after init):
		filename     - path to the PNG file.
		nx, ny       - image dimensions.
		PNG_bitdepth - detected or chosen bit depth (8 or 16).
	"""

	MAGIC = b'\x89PNG\r\n\x1a\n'

	# Class capability flags
	SUPPORT_STACK = False
	SUPPORT_3D = False
	SUPPORT_3D_STACK = False
	SUPPORT_COMPRESS = True

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode
		self._is_new_file = not os.path.exists(self.filename)
		self._initialized = False

		self.nx = None
		self.ny = None
		self.PNG_bitdepth = 16

		# Cached data for single-image format
		self._cached_meta = None
		self._cached_data = None

		if self._is_new_file:
			fmode = "wb"
		elif mode == "r":
			fmode = "rb"
		else:
			fmode = "rb+"

		try:
			self._file = open(self.filename, fmode)
		except OSError as e:
			raise FileIOError(f"Cannot open file '{self.filename}': {e}")

		# Read and cache the entire image on open
		if not self._is_new_file:
			self._parse_header()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None:
			self._file.close()
		return False

	def __len__(self):
		"""Return 1 — this format stores a single image."""
		return 1

	def _parse_header(self):
		"""Read PNG header and cache full image. Sets nx, ny, PNG_bitdepth."""
		with Image.open(self.filename) as img:

			# Convert color images to grayscale with a warning
			if img.mode not in ('L', 'I'):
				import warnings
				warnings.warn(f"PNG has mode '{img.mode}', converting to grayscale")
				img = img.convert('I')

			self.nx, self.ny = img.size

			# Detect bit depth from the PNG file's actual info
			try:
				# Pillow stores original bit depth in info for some configurations
				# A reliable heuristic: if max value > 255 it's 16-bit
				arr = np.asarray(img)
				if arr.max() > 255:
					self.PNG_bitdepth = 16
				else:
					self.PNG_bitdepth = 8
			except Exception:
				pass

			# Cache the full image data (float32 [0,1], Y-flipped for EMAN convention)
			if self.PNG_bitdepth == 16:
				data = np.asarray(img, dtype=np.float32) / 65535.0
			else:
				data = np.asarray(img, dtype=np.float32) / 255.0

			self._cached_data = data[::-1, :]

		self._initialized = True

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 8 = uint8, 16 = uint16."""
		return [8, 16]

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check whether a file is potentially a valid PNG image.

		Takes a filepath and optionally the first ~8 bytes of the file.
		PNG files start with the 8-byte signature 0x89 50 4E 47 0D 0A 1A 0A.
		"""
		try:
			if chunk is not None:
				return chunk[:8] == PngIO.MAGIC
			else:
				with open(os.path.expanduser(filepath), "rb") as f:
					return f.read(8) == PngIO.MAGIC
		except Exception:
			return False

	def read_header(self, index=0):
		"""Read the PNG header and return it as a dict.

		PNG is a single-image format; index must be 0 (or -1, treated as 0).

		Returns:
			dict with keys: nx, ny, nz (always 1), bitdepth.
		"""
		if index != 0 and index != -1:
			raise FileFormatError("PNG is a single-image format; index must be 0")

		if not self._initialized:
			self._parse_header()

		return {
			"nx": self.nx,
			"ny": self.ny,
			"nz": 1,
			"bitdepth": self.PNG_bitdepth,
		}

	def read_data(self, index=0):
		"""Read the PNG image data.

		Returns a numpy array of shape (ny, nx) with dtype float32 scaled to [0.0, 1.0].
		8-bit PNGs are divided by 255.0; 16-bit PNGs by 65535.0.
		Index must be 0.

		The Y-axis is flipped to match EMAN convention (row 0 at bottom).

		Returns:
			numpy.ndarray[float32] shaped (ny, nx).
		"""
		if index != 0 and index != -1:
			raise FileFormatError("PNG is a single-image format; index must be 0")

		return self._cached_data

	def write_header(self, meta, index=-1):
		"""Write a PNG header.

		meta must contain "nx" and "ny". Optional: "bitdepth" (default 16).
		PNG stores only 2D images; nz must be 1 if present.
		Index is ignored (single-image format), but returns 0.

		Returns:
			0 (the image index).
		"""
		if self.mode == "r":
			raise FileIOError("PNG file opened read-only; writes forbidden")

		nx = int(meta["nx"])
		ny = int(meta["ny"])
		nz = int(meta.get("nz", 1))

		if nz != 1:
			raise FileFormatError(f"Cannot write 3D image as PNG (nz={nz})")

		self.nx = nx
		self.ny = ny

		# Validate bitdepth
		bitdepth = int(meta.get("bitdepth", 16))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"PNG supports only 8 or 16-bit; got {bitdepth}")
		self.PNG_bitdepth = bitdepth

		# Clamp to valid values (fallback, should already be validated above)
		if self.PNG_bitdepth not in (8, 16):
			if self.PNG_bitdepth > 8:
				self.PNG_bitdepth = 16
			else:
				self.PNG_bitdepth = 8

		self._is_new_file = False
		self._initialized = True

		return 0

	def write_data(self, data, index=0):
		"""Write PNG image data.

		data should be a numpy array of shape (ny, nx) with dtype convertible
		to grayscale. Header must have been written first. Values are scaled
		and clipped to the appropriate range for the chosen bit depth.

		8-bit: [0, 255]. 16-bit: [0, 65535].

		The Y-axis is flipped to match EMAN convention (row 0 at bottom in
		memory, top in the PNG file).
		"""
		data = np.asarray(data)

		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]

		if data.ndim != 2:
			raise InvalidDimensions(f"Data must be 2D (ny, nx), got shape {data.shape}")

		ny, nx = data.shape
		if nx != self.nx or ny != self.ny:
			raise InvalidDimensions(f"Data shape ({ny}, {nx}) != header ({self.ny}, {self.nx})")

		# Flip Y to match EMAN convention (row 0 at bottom in memory, top in PNG file)
		flipped = data[::-1, :]

		# Scale to the target bit depth
		dmin, dmax = float(flipped.min()), float(flipped.max())
		if self.PNG_bitdepth == 16:
			max_range = 65535.0
			out_dtype = np.uint16
		else:
			max_range = 255.0
			out_dtype = np.uint8

		if dmax > dmin:
			out_data = np.clip(((flipped - dmin) / (dmax - dmin)) * max_range, 0, int(max_range)).astype(out_dtype)
		else:
			out_data = np.zeros((self.ny, self.nx), dtype=out_dtype)

		img = Image.fromarray(out_data, mode='I;16' if self.PNG_bitdepth == 16 else 'L')
		self._file.seek(0)
		img.save(self._file, format='PNG', bits=self.PNG_bitdepth)
		img.close()
