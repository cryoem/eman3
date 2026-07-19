#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ PgmIO)
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
# Foundation, Inc. 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""PGM (Portable Gray Map) image format reader/writer for EMAN.

PGM is a simple netpbm format. We only support binary PGM (P5 magic number).
A PGM file contains exactly one 2D image.

A "magic number" (two characters "P5")
Whitespace (blanks, TABs, CRs, LFs).
A width, formatted as ASCII characters in decimal.
Whitespace.
A height, again in ASCII decimal.
Whitespace.
The maximum gray value (Maxval), again in ASCII decimal. Must be less than 65536, and more than zero.
A single whitespace character (usually a newline).
A raster of Height rows, in order from top to bottom. Each row consists of Width gray values, in order from left to right. Each gray value is a number from 0 through Maxval, with 0 being black and Maxval being white. Each gray value is represented in pure binary by either 1 or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, it is 2 bytes. The most significant byte is first.

References: https://netpbm.sourceforge.io/doc/pgm.html
"""

import os
import numpy as np
import re
import time

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


class PgmIO:
	"""Read and write binary PGM (P5) format image files.

	PGM stores a single 2D grayscale image as uint8 pixel data with a simple
	ASCII header. Only the binary P5 variant is supported.

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with PgmIO("myfile.pgm", "r") as pgm:
			meta = pgm.read_header()
			data = pgm.read_data(0)   # shape (ny, nx), uint8

		with PgmIO("out.pgm", "rw") as pgm:
			pgm.write_header({"nx": 128, "ny": 128})
			pgm.write_data(image_uint8, 0)

	Attributes (read-only after init):
		filename  - path to the PGM file.
		nx, ny    - image dimensions from the header.
		maxval    - maximum gray value from the header.
	"""

	# Class capability flags
	SUPPORT_STACK = False
	SUPPORT_3D = False
	SUPPORT_3D_STACK = False
	SUPPORT_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode
		self._file = None

		self.nx = None
		self.ny = None
		self.nz = None
		self.maxval = None
		self.data_offset = None

		if not os.path.exists(self.filename) and mode=="r": raise FileNotFoundError(self.filename)

	# def __enter__(self):
	# 	return self

	# def __exit__(self, exc_type, exc_val, exc_tb):
	# 	if self._file is not None:
	# 		self._file.close()
	# 	return False

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 8 = uint8, 16 = uint16."""
		return [8, 16]

	@staticmethod
	def is_valid(filepath, chunk):
		"""Check whether a file is potentially a valid PGM image.

		Takes a filepath and the first ~1K of the file as bytes.
		If chunk is not provided, reads the file header itself.
		"""

		if chunk[:2]=="P5" and str.isspace(chunk[2]) : return True
		
		return False

	def read_header(self, index=0):
		"""Read the PGM header and return it as a dict.

		PGM is a single-image format; index must be 0 (or -1, treated as 0).

		Returns:
			dict with keys: nx, ny, nz (always 1), maxval.
		"""

		if index != 0: raise InvalidIndex("PGM is a single-image format; index must be 0")

		self._file=open(self.filename,"rb")
		raw = self._file.read(1024)		# we assume there won't be an unusually long comment line
		m = re.match(rb'^P5\s+(?:#.+\r?\n)*\s*(\d+)\s+(\d+)\s+(\d+)(\s)', raw)
		if not m:
		    raise FileFormatError("Invalid PGM header")
		
		self.nx = int(m.group(1))
		self.ny = int(m.group(2))
		self.maxval = int(m.group(3))
		# Detect bit depth from maxval (PGM spec: <256 means 8-bit, else 16-bit)
		self.bitdepth = 8 if self.maxval < 256 else 16
		# The trailing whitespace (group 4) is the mandatory single char before binary data
		self.data_offset = m.end()

		return { "nx": self.nx, "ny": self.ny, "nz": 1, "bitdepth": self.bitdepth }

	def read_data(self, index=0):
		"""Read the uint8 image data.

		Returns a numpy array of shape (ny, nx) with dtype uint8.
		Index must be 0.

		Returns:
			numpy.ndarray[uint8] shaped (ny, nx).

		WARNING: PGM specs require 16 bit files to be big-endian, but some programs use native endianness instead! We do not fix these
		"""
		if index != 0: raise InvalidIndex("PGM is a single-image format; index must be 0")

		self._file.seek(self.data_offset)
		if self.maxval<256 : 
			raw = self._file.read(self.nx * self.ny)
			if len(raw) < self.nx * self.ny:
				raise FileIOError(f"Incomplete data read: got {len(raw)} of {self.nx * self.ny} bytes")
		else: 
			raw=self._file.read(self.nx*self.ny*2)	# 16 bit unsigned file
			if len(raw) < self.nx * self.ny * 2:
				raise FileIOError(f"Incomplete data read: got {len(raw)} of {self.nx * self.ny * 2} bytes")

		self._file=None

		return np.frombuffer(raw, dtype=np.uint8).reshape((self.ny, self.nx), order='C')

	def write_header(self, meta, index=0):
		"""Write a PGM header.

		meta must contain "nx" and "ny". 
		if PGM.maxval is passed as >255 (65535), 16 bit files can be written
		PGM stores only 2D images; nz must be 1 if present.
		Index is ignored (single-image format), but returns 0.

		Returns:
			0 (the image index).
		"""
		if self.mode == "r":
			raise FileIOError("PGM file opened read-only; writes forbidden")

		self.nx = int(meta["nx"])
		self.ny = int(meta["ny"])
		self.nz = int(meta.get("nz", 1))

		if self.nz != 1:
			raise FileFormatError(f"Cannot write 3D image as PGM (nz={self.nz})")

		# Validate bitdepth and set maxval accordingly
		bitdepth = int(meta.get("bitdepth", 8))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"PGM supports only 8 or 16-bit; got {bitdepth}")
		self.bitdepth = bitdepth
		self.maxval = 255 if bitdepth == 8 else 65535

		self._file=open(self.filename,"wb")
		self._file.write(f"P5\n# Written by EMAN3 {time.ctime()}\n{self.nx} {self.ny}\n{self.maxval}\n".encode("ascii"))
		self.data_offset = self._file.tell()

		return 0

	def write_data(self, data, index=0):
		"""Write uint8 image data.

		data should be a numpy array of shape (ny, nx) with dtype convertible
		to uint8 or 16. Header must have been written first. Float values are
		scale to map (min,max) -> (0,255), for other mapping provide data as an integer array

		The C++ implementation flips the Y-axis on write (row 0 at top in file,
		bottom in memory convention). This is preserved here.

		"""
		if data.ndim == 3 and data.shape[0] == 1: data = data[0]
		if data.ndim != 2:
			raise InvalidDimensions(f"Data must be 2D (ny, nx), got shape {data.shape}")
		if data.shape[1] != self.nx or data.shape[0] != self.ny:
			raise InvalidDimensions(f"Data shape ({ny}, {nx}) != header ({self.ny}, {self.nx})")

		# y=0 is at the top in the file, bottom in numpy
		#flipped = data[::-1, :] #this is incorrect, not sure why based on specs
		flipped=data
		
		if data.dtype==np.uint8:
			self._file.write(flipped.ravel(order='C').tobytes())
		elif data.dtype==np.uint16:
			self._file.write(flipped.ravel(order='C').astype(">H").tobytes())	# > critical for big-endian
		if self.bitdepth == 8:
			data=(255.0*(flipped-flipped.min())/(flipped.max()-flipped.min())).astype("B")
			self._file.write(flipped.ravel(order='C').tobytes())
		else:
			data=(65535.0*(flipped-flipped.min())/(flipped.max()-flipped.min()))
			self._file.write(flipped.ravel(order='C').astype(">H").tobytes())	# > critical for big-endian

		self._file=None
