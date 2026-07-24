#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ JpegIO)
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

"""JPEG image format reader/writer for EMAN.

JPEG is a lossy compressed image format. Only 8-bit grayscale (mode 'L') images
are supported. Single image per file — no stacks.

Read support is provided even though the C++ was write-only, since Pillow makes
it trivial and round-trip testing is useful.

References: https://www.jpeg.org/
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


class JpegIO:
	"""Read and write JPEG format image files.

	JPEG stores a single 2D grayscale image as uint8 pixel data in lossy
	compressed form. Quality parameter controls compression level (1-100).

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with JpegIO("out.jpg", "rw") as jpeg:
			jpeg.write_header({"nx": 256, "ny": 256, "JPEG.quality": 90})
			jpeg.write_data(image_float, 0)

		with JpegIO("in.jpg", "r") as jpeg:
			meta = jpeg.read_header()
			data = jpeg.read_data(0)   # shape (ny, nx), float32

	Attributes (read-only after init):
		filename    - path to the JPEG file.
		nx, ny      - image dimensions.
		JPEG.quality - quality setting (used on write, read from EXIF on read).
	"""

	MAGIC = b'\xff\xd8\xff'

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
		self.JPEG_quality = 75

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

		# Read and cache the entire image on header read
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
		"Read JPEG header info using Pillow. Sets nx, ny, JPEG.quality."
		with Image.open(self.filename) as img:

			# Convert color images to grayscale with a warning
			if img.mode != 'L':
				import warnings
				warnings.warn(f"JPEG has mode '{img.mode}', will be converted to grayscale on read")
				img = img.convert('L')

			self.nx, self.ny = img.size

			# Try to extract quality from EXIF if available
			try:
				exif = img._getexif()
				if exif:
					for tag, val in exif.items():
						if str(val) == 'JPEGQuality':
							self.JPEG_quality = int(val.split(':')[0])
			except Exception:
				pass

			# Cache the full image data (float32 [0,1], Y-flipped for EMAN convention)
			data = np.asarray(img, dtype=np.float32) / 255.0
			self._cached_data = data[::-1, :]

		self._initialized = True

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 8 = uint8."""
		return [8]

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check whether a file is potentially a valid JPEG image.

		Takes a filepath and optionally the first ~1K of the file as bytes.
		JPEG files start with 0xFF 0xD8 0xFF (SOI marker).
		"""
		try:
			if chunk is not None:
				return chunk[:3] == JpegIO.MAGIC
			else:
				with open(os.path.expanduser(filepath), "rb") as f:
					return f.read(3) == JpegIO.MAGIC
		except Exception:
			return False

	def read_header(self, index=0):
		"""Read the JPEG header and return it as a dict.

		JPEG is a single-image format; index must be 0 (or -1, treated as 0).

		Returns:
			dict with keys: nx, ny, nz (always 1), bitdepth, JPEG.quality.
		"""
		if index != 0 and index != -1:
			raise FileFormatError("JPEG is a single-image format; index must be 0")

		if not self._initialized:
			self._parse_header()

		return {
			"nx": self.nx,
			"ny": self.ny,
			"nz": 1,
			"bitdepth": 8,
			"JPEG.quality": self.JPEG_quality,
		}

	def read_data(self, index=0):
		"""Read the JPEG image data.

		Returns a numpy array of shape (ny, nx) with dtype float32.
		JPEG stores uint8 [0,255]; values are scaled to float32 [0.0, 1.0].
		Index must be 0.

		The Y-axis is flipped to match EMAN convention (row 0 at bottom).

		Returns:
			numpy.ndarray[float32] shaped (ny, nx).
		"""
		if index != 0 and index != -1:
			raise FileFormatError("JPEG is a single-image format; index must be 0")

		return self._cached_data

	def write_header(self, meta, index=-1):
		"""Write a JPEG header.

		meta must contain "nx" and "ny". Optional: "bitdepth" (ignored, JPEG is 8-bit only),
		"JPEG.quality" (default 75). JPEG stores only 2D images; nz must be 1 if present.
		Index is ignored (single-image format), but returns 0.

		Returns:
			0 (the image index).
		"""
		if self.mode == "r":
			raise FileIOError("JPEG file opened read-only; writes forbidden")

		nx = int(meta["nx"])
		ny = int(meta["ny"])
		nz = int(meta.get("nz", 1))

		if nz != 1:
			raise FileFormatError(f"Cannot write 3D image as JPEG (nz={nz})")

		bitdepth = int(meta.get("bitdepth", 8))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"JPEG only supports 8-bit; got bitdepth={bitdepth}")

		self.nx = nx
		self.ny = ny
		self.JPEG_quality = int(meta.get("JPEG.quality", 75))

		self._is_new_file = False
		self._initialized = True

		return 0

	def write_data(self, data, index=0):
		"""Write uint8 JPEG image data.

		data should be a numpy array of shape (ny, nx) with dtype convertible
		to grayscale uint8. Header must have been written first. Values are
		scaled and clipped to [0, 255].

		The Y-axis is flipped to match EMAN convention (row 0 at bottom in
		memory, top in the JPEG file).
		"""
		data = np.asarray(data)

		if data.dtype != np.uint8:
			dmin, dmax = float(data.min()), float(data.max())
			if dmax > dmin:
				data = np.clip(((data - dmin) / (dmax - dmin)) * 255.0, 0, 255).astype(np.uint8)
			else:
				data = np.zeros((self.ny, self.nx), dtype=np.uint8)

		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]

		if data.ndim != 2:
			raise InvalidDimensions(f"Data must be 2D (ny, nx), got shape {data.shape}")

		ny, nx = data.shape
		if nx != self.nx or ny != self.ny:
			raise InvalidDimensions(f"Data shape ({ny}, {nx}) != header ({self.ny}, {self.nx})")

		# Flip Y to match EMAN convention (row 0 at bottom in memory, top in JPEG file)
		flipped = data[::-1, :]

		img = Image.fromarray(flipped, mode='L')
		self._file.seek(0)
		img.save(self._file, format='JPEG', quality=self.JPEG_quality)
		img.close()

