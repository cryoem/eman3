#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ TiffIO)
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

"""TIFF image format reader/writer for EMAN.

TIFF (Tagged Image File Format) supports multi-page grayscale images at
8-bit integer, 16-bit integer, and 32-bit float depth with optional
compression (LZW, deflate, etc.). Uses tifffile library which provides
direct access to libtiff functionality for full control over IFD directories
and sequential page writing without buffering all pages in RAM.

References: https://libtiff.org/, https://github.com/cgohlke/tifffile
"""

import os
import struct
import numpy as np
import tifffile


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


# TIFF magic bytes (little-endian and big-endian variants)
TIF_MAGIC_LE = b'II'
TIF_MAGIC_BE = b'MM'


class TiffIO:
	"""Read and write multi-page TIFF format image files using tifffile.

	TIFF stores grayscale images at 8-bit, 16-bit or float32 depth with
	optional compression. Multiple images per file supported by writing one
	page at a time — pages are not buffered in RAM.

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with TiffIO("out.tif", "rw") as tif:
			for i in range(n_images):
				idx = tif.write_header(meta[i])
				tif.write_data(data[i], idx)

		with TiffIO("in.tif", "r") as tif:
			for i in range(tif.nimg):
				meta = tif.read_header(i)
				data = tif.read_data(i)

	Attributes (read-only after init):
		filename   - path to the TIFF file.
		nimg       - total number of images in the file.
		nx, ny     - dimensions from the first directory entry.
		big_endian - endianness of the on-disk data.
	"""

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode
		self._is_new_file = not os.path.exists(self.filename)
		self._initialized = False

		self.nx = None
		self.ny = None
		self.nimg = 0
		self.big_endian = False

		# Writer (kept open for sequential page writes, no copy-on-write)
		self._writer = None
		# Reader (kept open for reading directories/pages)
		self._tif = None

		if self.mode == "r":
			if not os.path.exists(self.filename):
				raise FileNotFoundError(f"TIFF file '{self.filename}' not found")
			self._tif = tifffile.TiffFile(self.filename)
			self._count_pages()
		else:
			# Write mode: writer opened lazily on first write_data call
			if self._is_new_file:
				self.nimg = 0
				self._initialized = True
			else:
				# Count existing pages so we know how many to append after
				with tifffile.TiffFile(self.filename) as src:
					pages = list(src.pages)
					self.nimg = len(pages)
					if self.nimg > 0:
						self.nx, self.ny = pages[0].shape
					# Detect endianness from file header
					with open(self.filename, "rb") as f:
						header = f.read(2)
						self.big_endian = (header == b'MM')
				self._initialized = True

	def _count_pages(self):
		"""Count pages by reading directory entries."""
		if self._tif is None:
			return

		try:
			pages = list(self._tif.pages)
			self.nimg = len(pages)
			if self.nimg > 0:
				self.nx, self.ny = pages[0].shape
				# Detect endianness from file header
				with open(self.filename, "rb") as f:
					header = f.read(2)
					self.big_endian = (header == b'MM')
		except Exception:
			self.nimg = 0

		self._initialized = True

	def __del__(self):
		if self._writer:
			try: self._writer.close()
			except Exception: pass
		if self._tif:
			try: self._tif.close()
			except Exception: pass

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 0=float32, 8=uint8, 16=uint16."""
		return [0, 8, 16]

	@staticmethod
	def is_valid(filepath, chunk):
		"""Check whether a file is potentially a valid TIFF image.

		TIFF files start with 'II\\x2a\\x00' (little-endian) or 'MM\\x00\\x2a' (big-endian).
		"""
		try:
			if len(chunk) < 4:
				return False
			if chunk[:2] == TIF_MAGIC_LE and struct.unpack('<H', chunk[2:4])[0] in (42, 43):
				return True
			if chunk[:2] == TIF_MAGIC_BE and struct.unpack('>H', chunk[2:4])[0] in (42, 43):
				return True
			return False
		except Exception:
			return False

	def _get_numpy_dtype(self, bitdepth):
		"""Map bitdepth to numpy dtype."""
		if bitdepth == 8:
			return np.uint8
		elif bitdepth == 16:
			return np.uint16
		else:
			return np.float32

	def _detect_bitdepth_from_page(self, page):
		"""Detect bit depth from tifffile Page object."""
		dt = page.dtype
		if dt == np.uint8:
			return 8
		elif dt == np.uint16:
			return 16
		else:
			return 0

	def read_header(self, index):
		"""Read the TIFF header for one image and return it as a dict.

		TIFF is a multi-image format; each page is an independent IFD directory.
		Returns a Python dict with nx, ny, nz (always 1), bitdepth, nimg, etc.

		Parameters:
			index : int  Image number (0-based).

		Returns:
			dict containing header metadata.
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		page = self._tif.pages[index]
		ny, nx = page.shape
		bitdepth = self._detect_bitdepth_from_page(page)

		if index == 0:
			self.nx, self.ny = nx, ny

		return {
			"nx": nx,
			"ny": ny,
			"nz": 1,
			"bitdepth": bitdepth,
			"nimg": self.nimg,
			"TIFF.compression": page.compression.name if page.compression else "raw",
		}

	def read_data(self, index):
		"""Read the TIFF image data.

		Returns a numpy array of shape (ny, nx) with dtype float32 scaled to [0,1].
		Integer images are divided by their max range before returning.

		The Y-axis is flipped to match EMAN convention (row 0 at bottom).

		Parameters:
			index : int  Image number (0-based).

		Returns:
			numpy.ndarray[float32] shaped (ny, nx).
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		page = self._tif.pages[index]
		data = page.asarray()
		bitdepth = self._detect_bitdepth_from_page(page)

		# Scale to [0, 1] for integer types
		if bitdepth == 8:
			data = data.astype(np.float32) / 255.0
		elif bitdepth == 16:
			data = data.astype(np.float32) / 65535.0
		else:
			data = data.astype(np.float32)

		# Flip Y to match EMAN convention (row 0 at bottom in memory)
		return data[::-1, :]

	def write_header(self, meta, index=-1):
		"""Write a TIFF header entry.

		meta must contain "nx" and "ny". Optional: "bitdepth" (default 0=float32),
		"TIFF.compression" (default "raw").
		TIFF stores only 2D images; nz must be 1 if present.

		If index == -1 (default), appends a new image to the file.
		Returns the image index that was written.

		The actual TIFF file is only updated when write_data is called.
		"""
		if self.mode == "r":
			raise FileIOError("TIFF file opened read-only; writes forbidden")

		nx = int(meta["nx"])
		ny = int(meta["ny"])
		nz = int(meta.get("nz", 1))

		if nz != 1:
			raise FileFormatError(f"Cannot write 3D image as TIFF (nz={nz})")

		bitdepth = int(meta.get("bitdepth", 0))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"TIFF supports only 8-bit, 16-bit or float32; got {bitdepth}")

		self.nx = nx
		self.ny = ny
		self._write_bitdepth = bitdepth
		self._write_compression_name = meta.get("TIFF.compression", "raw")

		if self._is_new_file:
			self._is_new_file = False
			self.nimg = 0
			self._initialized = True

		if index < 0:
			index = self.nimg
		self.nimg = max(self.nimg, index + 1)

		return index

	def write_data(self, data, index):
		"""Write TIFF image data.

		data should be a numpy array of shape (ny, nx). Float32 input is
		written directly for bitdepth=0; otherwise scaled to the target range.
		Integer arrays are written directly without scaling.

		The Y-axis is flipped to match EMAN convention (row 0 at bottom in
		memory, top in the TIFF file).

		A single persistent TiffWriter stays open across calls and appends pages
		sequentially — no copy-on-write or RAM buffering of existing pages.
		BigTIFF is used automatically for multi-GB files.
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		data = np.asarray(data)

		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]

		if data.ndim != 2:
			raise InvalidDimensions(f"Data must be 2D (ny, nx), got shape {data.shape}")

		ny, nx = data.shape
		if nx != self.nx or ny != self.ny:
			raise InvalidDimensions(f"Data shape ({ny}, {nx}) != header ({self.ny}, {self.nx})")

		bitdepth = self._write_bitdepth
		out_dtype = self._get_numpy_dtype(bitdepth)

		# Flip Y to match EMAN convention (row 0 at bottom in memory, top in TIFF)
		flipped = data[::-1, :]

		# Convert to target dtype
		if flipped.dtype != out_dtype:
			if bitdepth == 0:
				out_data = flipped.astype(np.float32)
			elif data.dtype.kind not in ('u', 'i'):
				dmin, dmax = float(flipped.min()), float(flipped.max())
				max_range = 255 if bitdepth == 8 else 65535
				if dmax > dmin:
					out_data = np.clip(((flipped - dmin) / (dmax - dmin)) * max_range, 0, max_range).astype(out_dtype)
				else:
					out_data = np.zeros((self.ny, self.nx), dtype=out_dtype)
			else:
				out_data = flipped.astype(out_dtype)
		else:
			out_data = flipped

		# Map compression name to tifffile constant
		comp_map = {
			"raw": None,
			"tiff_lzw": "lzw",
			"tiff_deflate": "zlib",
			"tiff_adobe_deflate": "zlib",
			"packbits": "packbits",
		}
		comp_val = comp_map.get(self._write_compression_name, None)

		# Fallback to zlib if lzw is not available
		if comp_val == "lzw":
			try:
				import imagecodecs  # noqa: F401
			except ImportError:
				comp_val = "zlib"

		# Open persistent writer on first call; keep it open for all subsequent writes
		if self._writer is None:
			if self._is_new_file or self.nimg == 0:
				self._writer = tifffile.TiffWriter(self.filename, bigtiff=True)
			else:
				self._writer = tifffile.TiffWriter(self.filename, append=True, bigtiff=True)

		# Append this page directly to disk (no copy-on-write)
		self._writer.write(out_data, compression=comp_val)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._writer:
			try: self._writer.close()
			except Exception: pass
		if self._tif:
			try: self._tif.close()
			except Exception: pass
		self._writer = None
		self._tif = None
		return False
