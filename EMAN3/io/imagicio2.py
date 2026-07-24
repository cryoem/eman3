#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ ImagicIO2)
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

"""IMAGIC-4D file format reader/writer for EMAN.

IMAGIC stores paired .hed (header) and .img (data) files. Each image has
a fixed-size header record in the .hed file, and raw data in the .img file.

Header layout (Imagic4D struct, one per image):
	int imgnum    (1-based image index)
	int count     (total images - 1; first record only)
	int error
	int headrec   (always 1)
	int month
	int mday
	int year
	int hour
	int minute
	int sec
	int rsize     (image size in bytes, or -1 if overflow)
	int izold
	int ny        (number of lines per image)
	int nx        (pixels per line)
	char[4] type  ('REAL' for float32, 'INTG' for int16)
	int ixold
	int iyold
	float avdens  (average density)
	float sigma
	... (many more fields: Euler angles, CTF params, labels, etc.)

Data layout: contiguous raw data in .img file. REAL=float32, INTG=int16.
Multiple images stored sequentially. Each image can be 2D or 3D.

References: http://www.imagescience.de/formats/formats.htm
"""

import os
import struct
import time
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


# IMAGIC-4D header struct as a numpy dtype. Each field matches the C++
# Imagic4D struct exactly, including padding for char arrays.
IMAGIC_HEADER_DTYPE = np.dtype([
	("imgnum",       "<i4"),     # 0: 1-based image index
	("count",        "<i4"),     # 1: total images - 1 (first record)
	("error",        "<i4"),     # 2: error code
	("headrec",      "<i4"),     # 3: always 1
	("month",        "<i4"),     # 5: creation month (0-based in tm struct)
	("mday",         "<i4"),     # 6: creation day
	("year",         "<i4"),     # 7: creation year
	("hour",         "<i4"),     # 8: creation hour
	("minute",       "<i4"),     # 9: creation minute
	("sec",          "<i4"),     # 10: creation second
	("rsize",        "<i4"),     # 11: image size in bytes
	("izold",        "<i4"),     # 12: top-left Z coord before cut
	("ny",           "<i4"),     # 13: number of lines per image
	("nx",           "<i4"),     # 14: pixels per line
	("type",         "S4"),      # 15: 'REAL', 'INTG', 'PACK', 'COMP'
	("ixold",        "<i4"),     # 16: top-left X coord before cut
	("iyold",        "<i4"),     # 17: top-left Y coord before cut
	("avdens",       "<f4"),     # 18: average density
	("sigma",        "<f4"),     # 19: standard deviation
	("user1",        "<f4"),     # 20: user-defined
	("user2",        "<f4"),     # 21: user-defined
	("densmax",      "<f4"),     # 22: highest density
	("densmin",      "<f4"),     # 23: minimal density
	("complex_flag", "<i4"),     # 24: complex data flag
	("defocus1",     "<f4"),     # 25: defocus value 1
	("defocus2",     "<f4"),     # 26: defocus value 2
	("defangle",     "<f4"),     # 27: defocus angle
	("sinostart",    "<f4"),     # 28: sinogram start angle
	("sinoend",      "<f4"),     # 29: sinogram end angle
	("label",        "S80"),     # 30-49: image name/title (80 chars)
	("ccc3d",        "<f4"),     # 50: 3D similarity criteria
	("ref3d",        "<i4"),     # 51: 3D membership
	("mident",       "<i4"),     # 52: micrograph identification
	("ezshift",      "<i4"),     # 53: equivalent shift in Z
	("ealpha",       "<i4"),     # 54: equivalent Euler alpha
	("ebeta",        "<i4"),     # 55: equivalent Euler beta
	("egamma",       "<i4"),     # 56: equivalent Euler gamma
	("unused1",      "<i4"),     # 57
	("unused2",      "<i4"),     # 58
	("nalisum",      "<i4"),     # 59: number of images summed
	("pgroup",       "<i4"),     # 60: point-group symmetry
	("izlp",         "<i4"),     # 61: Z dimension (number of sections)
	("i4lp",         "<i4"),     # 62: number of objects in file
	("i5lp",         "<i4"),     # 63
	("i6lp",         "<i4"),     # 64
	("alpha",        "<f4"),     # 65: Euler angle alpha
	("beta",         "<f4"),     # 66: Euler angle beta
	("gamma",        "<f4"),     # 67: Euler angle gamma
	("imavers",      "<i4"),     # 68: IMAGIC version (yyyymmdd)
	("realtype",     "<i4"),     # 69: machine stamp
	("buffer",       "S120"),    # 70-99: buffering control (do not touch)
	("angle",        "<f4"),     # 100: last rotation angle
	("voltage",      "<f4"),     # 101: acceleration voltage (kV)
	("spaberr",      "<i4"),     # 102: spherical aberration (mm)
	("pcoher",       "<i4"),     # 103: partial coherence
	("ccc",          "<f4"),     # 104: cross-correlation peak height
	("errar",        "<f4"),     # 105: error in angular reconstitution
	("err3d",        "<f4"),     # 106: error in 3D reconstruction
	("ref",          "<i4"),     # 107: reference number
	("classno",      "<f4"),     # 108: class number
	("locold",       "<f4"),     # 109: location before cut
	("repqual",      "<f4"),     # 110: representation quality
	("zshift",       "<f4"),     # 111: shift in Z
	("xshift",       "<f4"),     # 112: shift in X
	("yshift",       "<f4"),     # 113: shift in Y
	("numcls",       "<f4"),     # 114: number of members in class
	("ovqual",       "<f4"),     # 115: overall class quality
	("eangle",       "<f4"),     # 116: equivalent angle
	("exshift",      "<f4"),     # 117: equivalent shift X
	("eyshift",      "<f4"),     # 118: equivalent shift Y
	("cmtotvar",     "<f4"),     # 119: total variance relative to center of mass
	("informat",     "<f4"),     # 120: Gauss norm / real*FT space info
	("numeigen",     "<i4"),     # 121: number of eigenvalues in MSA
	("niactive",     "<i4"),     # 122: number of active images in MSA
	("resolx",       "<f4"),     # 123: Angstrom/pixel X
	("resoly",       "<f4"),     # 124: Angstrom/pixel Y
	("resolz",       "<f4"),     # 125: Angstrom/pixel Z
	("alpha2",       "<f4"),     # 126: Euler alpha (projection matching)
	("beta2",        "<f4"),     # 127: Euler beta (projection matching)
	("gamma2",       "<f4"),     # 128: Euler gamma (projection matching)
	("nmetric",      "<f4"),     # 129: metric used in MSA
	("actmsa",       "<f4"),     # 130: active flag for MSA
	("coosmsa",      "<f4", (69,)), # 131-199: factorial axis coordinates (MSA)
	("history",      "S228"),    # 220-256: coded history (228 chars)
])

# Machine stamp constants for realtype field
REALTYPE_VAX_VMS = 16777216
REALTYPE_LINUX_WINDOWS = 33686018
REALTYPE_SGI_IBM = 67372036


class ImagicIO:
	"""Read and write IMAGIC-4D paired .hed/.img files.

	IMAGIC stores one header record per image in the .hed file, and raw
	float32 data in the .img file. Supports stacks of 2D or 3D images.

	Modes: "r" (read-only), "rw" (read-write, creates new files if they
	don't exist). Both .hed and .img must exist (or both must not exist).

	Usage example::

		with ImagicIO("myfile", "rw") as im:
			idx = im.write_header({"nx": 256, "ny": 256, "nz": 1}, -1)
			im.write_data(data, idx)

		with ImagicIO("myfile", "r") as im:
			for i in range(im.nimg):
				meta = im.read_header(i)
				data = im.read_data(i)   # shape (nz, ny, nx), float32

	Attributes (read-only after init):
		filename    - base filename (without extension).
		nx, ny, nz  - image dimensions.
		nimg        - number of images in the stack.
	"""

	# Class capability flags
	SUPPORT_STACK = True
	SUPPORT_3D = True
	SUPPORT_3D_STACK = False
	SUPPORT_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)

		# Strip extension if present, derive paired filenames
		base, ext = os.path.splitext(self.filename)
		if ext.lower() in (".hed", ".img"):
			self.hed_filename = base + ".hed"
			self.img_filename = base + ".img"
		else:
			self.hed_filename = self.filename + ".hed"
			self.img_filename = self.filename + ".img"

		self.mode = mode
		self._initialized = False
		self.big_endian = False

		self.nx = None
		self.ny = None
		self.nz = 1
		self.nimg = 0
		self.datatype = "REAL"

		# File handles (kept open for sequential access)
		self._hed_file = None
		self._img_file = None

		if mode == "r":
			if not os.path.exists(self.hed_filename):
				raise FileNotFoundError(f"IMAGIC header file '{self.hed_filename}' not found")
			if not os.path.exists(self.img_filename):
				raise FileNotFoundError(f"IMAGIC data file '{self.img_filename}' not found")
			self._hed_file = open(self.hed_filename, "rb")
			self._img_file = open(self.img_filename, "rb")
			self._parse_header(0)

		else:  # rw mode
			is_new_hed = not os.path.exists(self.hed_filename)
			is_new_img = not os.path.exists(self.img_filename)

			if is_new_hed != is_new_img:
				raise FileIOError("IMAGIC .hed and .img files must both exist or both be new")

			mode_flag = "w+b" if is_new_hed else "rb+"
			self._hed_file = open(self.hed_filename, mode_flag)
			self._img_file = open(self.img_filename, mode_flag)

			if is_new_hed:
				self.nimg = 0
				self._initialized = True
			else:
				# Count existing images by reading first header
				self._parse_header(0)
				self._initialized = True

	def __del__(self):
		"""Flush and close file handles on destruction."""
		for fh in (self._hed_file, self._img_file):
			if fh is not None:
				try: fh.flush()
				except Exception: pass
				try: fh.close()
				except Exception: pass

	def __len__(self):
		"""Return number of images in the file."""
		return self.nimg

	def _parse_header(self, index=0):
		"""Read and parse the Imagic4D header for one image.

		Detects endianness from realtype field. Populates nx, ny, nz, nimg.
		"""
		self._hed_file.seek(index * IMAGIC_HEADER_DTYPE.itemsize)
		raw = self._hed_file.read(IMAGIC_HEADER_DTYPE.itemsize)
		if len(raw) < IMAGIC_HEADER_DTYPE.itemsize:
			raise FileIOError(f"Incomplete IMAGIC header read at index {index}")

		hed = np.frombuffer(raw, dtype=IMAGIC_HEADER_DTYPE)[0]

		# Detect endianness from realtype field
		realtype = hed["realtype"]
		if int(realtype) in (REALTYPE_VAX_VMS, REALTYPE_LINUX_WINDOWS, REALTYPE_SGI_IBM):
			self.big_endian = False
		else:
			# Try big-endian interpretation
			be_realtype = np.frombuffer(raw[struct.calcsize("<i4") * 69:struct.calcsize("<i4") * 69 + 4], dtype=">i4")[0]
			if int(be_realtype) in (REALTYPE_VAX_VMS, REALTYPE_LINUX_WINDOWS, REALTYPE_SGI_IBM):
				self.big_endian = True
				# Re-parse as big-endian
				be_dtype = IMAGIC_HEADER_DTYPE.newbyteorder(">")
				hed = np.frombuffer(raw, dtype=be_dtype)[0]

		self.datatype = hed["type"].decode("ascii", errors="ignore").strip("\x00")
		# Read dimensions directly from header (no swap needed - C++ writes them straight)
		self.nx = int(hed["nx"])
		self.ny = int(hed["ny"])
		self.nz = max(int(hed["izlp"]), 1)
		# Only set nimg from first header (count field is only valid in header 0)
		if index == 0:
			self.nimg = max(int(hed["count"]) + 1, 1)

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 0=float32 (write default)."""
		return [0]

	@staticmethod
	def is_valid(filepath):
		"""Check whether a .hed file is a valid IMAGIC-4D header.

		Reads the first Imagic4D record and validates required fields:
		realtype must be one of the known machine stamps, headrec==1,
		dimensions in reasonable ranges, etc. Matches C++ SpiderIO::is_valid.
		"""
		hed_path = filepath if filepath.endswith(".hed") else filepath + ".hed"
		try:
			with open(hed_path, "rb") as f:
				raw = f.read(IMAGIC_HEADER_DTYPE.itemsize)
			if len(raw) < IMAGIC_HEADER_DTYPE.itemsize:
				return False

			hed = np.frombuffer(raw, dtype=IMAGIC_HEADER_DTYPE)[0]

			# Check realtype - try LE first, then BE
			realtype = int(hed["realtype"])
			if realtype not in (REALTYPE_VAX_VMS, REALTYPE_LINUX_WINDOWS, REALTYPE_SGI_IBM):
				be_dtype = IMAGIC_HEADER_DTYPE.newbyteorder(">")
				hed_be = np.frombuffer(raw, dtype=be_dtype)[0]
				realtype = int(hed_be["realtype"])
				if realtype not in (REALTYPE_VAX_VMS, REALTYPE_LINUX_WINDOWS, REALTYPE_SGI_IBM):
					return False
				hed = hed_be

			# Validate required integer fields and ranges
			headrec = int(hed["headrec"])
			count = int(hed["count"])
			nx = int(hed["nx"])
			ny = int(hed["ny"])
			nz = max(int(hed["izlp"]), 1)
			hour = int(hed["hour"])
			minute = int(hed["minute"])
			second = int(hed["sec"])

			max_dim = 1 << 20
			if headrec != 1:
				return False
			if count < 0 or count >= max_dim:
				return False
			if nx <= 0 or nx >= max_dim:
				return False
			if ny <= 0 or ny >= max_dim:
				return False
			if nz <= 0 or nz >= max_dim:
				return False
			if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
				return False

			return True

		except Exception:
			return False

	def read_header(self, index):
		"""Read the IMAGIC header for one image and return it as a dict.

		Parameters:
			index : int  Image number (0-based).

		Returns:
			dict with keys: nx, ny, nz, bitdepth, nimg, minimum, maximum,
			mean, sigma, etc.
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		self._parse_header(index)

		self._hed_file.seek(index * IMAGIC_HEADER_DTYPE.itemsize)
		raw = self._hed_file.read(IMAGIC_HEADER_DTYPE.itemsize)
		if len(raw) < IMAGIC_HEADER_DTYPE.itemsize:
			raise FileIOError(f"Incomplete IMAGIC header read at index {index}")

		byte_order = ">" if self.big_endian else "<"
		dtype_be = IMAGIC_HEADER_DTYPE.newbyteorder(byte_order)
		hed = np.frombuffer(raw, dtype=dtype_be)[0]

		result = {
			"nx": int(hed["nx"]),
			"ny": int(hed["ny"]),
			"nz": max(int(hed["izlp"]), 1),
			"nimg": int(hed["count"]) + 1,
			"bitdepth": 0 if hed["type"] == b"REAL" else 16,
			"minimum": float(hed["densmin"]),
			"maximum": float(hed["densmax"]),
			"mean": float(hed["avdens"]),
			"sigma": float(hed["sigma"]),
			"IMAGIC.imgnum": int(hed["imgnum"]),
			"IMAGIC.type": hed["type"].decode("ascii", errors="ignore").strip("\x00"),
			"IMAGIC.label": hed["label"].decode("ascii", errors="ignore").strip(),
			"IMAGIC.history": hed["history"].decode("ascii", errors="ignore").strip(),
		}

		if float(hed["defocus1"]) != 0.0 or float(hed["defocus2"]) != 0.0:
			result["euler_alpha"] = float(hed["alpha"])
			result["euler_beta"] = float(hed["beta"])
			result["euler_gamma"] = float(hed["gamma"])
			result["IMAGIC.defocus1"] = float(hed["defocus1"])
			result["IMAGIC.defocus2"] = float(hed["defocus2"])
			result["IMAGIC.defangle"] = float(hed["defangle"])

		return result

	def read_data(self, index):
		"""Read IMAGIC image data as float32.

		Returns a numpy array of shape (nz, ny, nx) with dtype float32.
		INTG (int16) data is converted to float32 on read.

		Parameters:
			index : int  Image number (0-based).

		Returns:
			numpy.ndarray[float32] shaped (nz, ny, nx).
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		self._parse_header(0)  # Ensure dims are loaded

		nx = self.nx
		ny = self.ny
		nz = self.nz
		total = nx * ny * nz

		data_offset = index * total * 4
		self._img_file.seek(data_offset)

		# Read data directly, then Y-flip and X-unshift to restore EMAN convention
		if self.datatype == "REAL":
			raw = self._img_file.read(total * 4)
			if len(raw) < total * 4:
				raise FileIOError(f"Incomplete data read at index {index}")
			byte_order = ">" if self.big_endian else "<"
			data = np.frombuffer(raw, dtype=byte_order + "f4").reshape((nz, ny, nx))

		elif self.datatype == "INTG":
			raw = self._img_file.read(total * 2)
			if len(raw) < total * 2:
				raise FileIOError(f"Incomplete data read at index {index}")
			byte_order = ">" if self.big_endian else "<"
			data = np.frombuffer(raw, dtype=byte_order + "i2").astype(np.float32).reshape((nz, ny, nx))

		else:
			raise FileFormatError(f"Unsupported IMAGIC data type: {self.datatype}")

		# Y-flip to restore EMAN convention
		data = data[:, ::-1, :]

		# Preserve dimensionality: if nz==1, return 2D array to match write input
		if self.nz == 1:
			return data[0]
		return data

	def write_header(self, meta, index=-1):
		"""Write an IMAGIC header entry.

		meta must contain "nx", "ny", "nz". Optional: "minimum", "maximum",
		"mean", "sigma", "euler_alpha/beta/gamma", "IMAGIC.label", etc.

		If index == -1, appends a new image.
		Returns the image index that was written.

		Only float32 (REAL) output is supported.
		"""
		if self.mode == "r":
			raise FileIOError("IMAGIC file opened read-only; writes forbidden")

		# Validate bitdepth (IMAGIC write is REAL/float32 only)
		bitdepth = int(meta.get("bitdepth", 0))
		if bitdepth != 0:
			raise FileFormatError(f"IMAGIC write only supports float32 (bitdepth=0); got {bitdepth}")

		nx = int(meta["nx"])
		ny = int(meta["ny"])
		nz = int(meta.get("nz", 1))

		# For new files, store dimensions. For existing files, validate consistency.
		if self.nx is None:
			self.nx = nx
			self.ny = ny
			self.nz = nz
		else:
			if (nx != self.nx or ny != self.ny or nz != self.nz):
				raise FileFormatError(
					f"IMAGIC dimensions {nx}x{ny}x{nz} mismatch existing "
					f"{self.nx}x{self.ny}x{self.nz}"
				)

		self.nz = nz

		if index < 0:
			index = self.nimg
		self.nimg = max(self.nimg, index + 1)

		# Build header record
		hed = np.zeros(1, dtype=IMAGIC_HEADER_DTYPE)[0]

		now = time.localtime()
		hed["imgnum"] = index + 1
		hed["count"] = self.nimg - 1
		hed["headrec"] = 1
		hed["month"] = now.tm_mon
		hed["mday"] = now.tm_mday
		hed["year"] = now.tm_year
		hed["hour"] = now.tm_hour
		hed["minute"] = now.tm_min
		hed["sec"] = now.tm_sec

		img_size = nx * ny * nz
		hed["rsize"] = img_size if img_size < 2**31 else -1
		hed["nx"] = nx         # width/pixels per line
		hed["ny"] = ny         # height/lines per image
		hed["type"] = b"REAL"
		hed["izlp"] = nz

		hed["avdens"] = float(meta.get("mean", 0.0))
		hed["sigma"] = float(meta.get("sigma", 0.0))
		hed["densmax"] = float(meta.get("maximum", 0.0))
		hed["densmin"] = float(meta.get("minimum", 0.0))

		if meta.get("euler_alpha") is not None: hed["alpha"] = float(meta["euler_alpha"])
		if meta.get("euler_beta") is not None:  hed["beta"] = float(meta["euler_beta"])
		if meta.get("euler_gamma") is not None: hed["gamma"] = float(meta["euler_gamma"])

		hed["realtype"] = REALTYPE_LINUX_WINDOWS
		hed["i4lp"] = self.nimg

		label = str(meta.get("IMAGIC.label", "EMAN3")).encode("ascii", errors="ignore")[:80]
		hed["label"] = label.ljust(80, b"\x00")

		history = str(meta.get("IMAGIC.history", "written by EMAN3")).encode("ascii", errors="ignore")[:228]
		hed["history"] = history.ljust(228, b"\x00")

		if meta.get("apix_x"): hed["resolx"] = float(meta["apix_x"])
		if meta.get("apix_y"): hed["resoly"] = float(meta["apix_y"])
		if meta.get("apix_z"): hed["resolz"] = float(meta["apiz_z"])

		# Write the per-image header record
		self._hed_file.seek(index * IMAGIC_HEADER_DTYPE.itemsize)
		self._hed_file.write(hed.tobytes())

		# Update first header with total image count
		self._hed_file.seek(0)
		first_raw = self._hed_file.read(IMAGIC_HEADER_DTYPE.itemsize)
		first_hed = np.frombuffer(bytearray(first_raw), dtype=IMAGIC_HEADER_DTYPE)[0]
		first_hed["count"] = self.nimg - 1
		first_hed["i4lp"] = self.nimg
		self._hed_file.seek(0)
		self._hed_file.write(first_hed.tobytes())

		if not self._initialized:
			self._initialized = True

		return index

	def write_data(self, data, index):
		"""Write float32 IMAGIC image data.

		data should be a numpy array of shape (nz, ny, nx) or (ny, nx).
		Header must have been written first (via write_header).

		Data is stored as float32 in the .img file.
		"""
		if index < 0 or index >= self.nimg:
			raise InvalidIndex(f"Image index {index} out of range [0, {self.nimg})")

		data = np.asarray(data, dtype=np.float32)

		if data.ndim == 2:
			nz = 1
			ny, nx = data.shape
		elif data.ndim == 3 and data.shape[0] == 1:
			data = data[0]
			nz = 1
			ny, nx = data.shape
		elif data.ndim == 3:
			nz, ny, nx = data.shape
		else:
			raise InvalidDimensions(f"Data must be 2D or 3D, got shape {data.shape}")

		if nx != self.nx or ny != self.ny or nz != self.nz:
			raise InvalidDimensions(
				f"Data shape ({nz},{ny},{nx}) != header ({self.nz},{self.ny},{self.nx})"
			)

		total = nx * ny * nz
		data_offset = index * total * 4
		self._img_file.seek(data_offset)
		# Flip Y for EMAN convention (row 0 at bottom in memory,
		# top in the IMAGIC file as per spec: first pixel = upper left)
		flipped = np.asarray(data, dtype=np.float32)
		if flipped.ndim == 2:
			flipped = flipped[::-1, :]
		else:
			flipped = flipped[:, ::-1, :]
		self._img_file.write(flipped.ravel(order="C").astype("<f4").tobytes())

	def flush(self):
		if self._hed_file: self._hed_file.flush()
		if self._img_file: self._img_file.flush()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Flush and close file handles on context exit."""
		for fh in (self._hed_file, self._img_file):
			if fh is not None:
				try: fh.flush()
				except Exception: pass
				try: fh.close()
				except Exception: pass
		self._hed_file = None
		self._img_file = None
		return False
