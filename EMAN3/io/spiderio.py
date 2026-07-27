#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ SpiderIO)
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""SPIDER image format reader/writer for EMAN.

SPIDER (System for Processing Image Data from Electron microscopy) stores
floating-point image data with a rich binary header. Two file flavors:

	Single image - one header followed by image data.
	Image stack	 - an overall header followed by per-image headers+data.

Header layout (sizeof(SpiderHeader) = 1023 bytes):
	Bytes 0-843:   211 float-sized slots (37 named floats + u8[48] padding +
				  xf[27] + u9[135]). The entire 844-byte region is treated
				  as float data for byte-order swapping.
	Bytes 844-854: date[11]	 (char, e.g. "27-MAY-1999")
	Bytes 855-862: time[8]	 (char, e.g. "09:43:19")
	Bytes 863-1022: title[160] (char)
The header is padded on-disk to be an integer multiple of nx*4 bytes
(record length).

Data layout: flat float32 array of size nz*ny*nx in z-fastest order
(Fortran-style), i.e., shape (nz, ny, nx) in NumPy convention.

References: http://www.wadsworth.org/spider_doc/spider/docs/index.html
"""

import sys
import os
import struct
from math import *
import numpy as np
from datetime import datetime

#from EMAN3.EMAN3 import FileIOError
class FileFormatError(Exception):
	"""Exception for problems with File formats"""

class FileIOError(Exception):
	"""Exception for problems with File IO"""

class InvalidDimensions(Exception):
	"""Indicates a mismatch between data size and header indicated size"""

# ---------------------------------------------------------------------------
# SPIDER header constants
# ---------------------------------------------------------------------------

SPIDER_FLOATS_IN_HEADER = 211				   # float-sized slots in the numeric portion
SPIDER_HEADER_SIZE = 1024						# technically 1023, but we put a zero in the last byte, and since record length is a multiple of 4, calling it 1024 is fine

# float fields by name and float location
_HEADER_FLOATS = {
	"nz":0, 
	"ny":1, 
	"SPIDER.type":4,	# 1 - 2D, 3 - 3D, -11 - 2D FFT odd, -12 - 2D FFT even, -21 - 3D FFT odd, -22 - 3D FFT even (EMAN3 doesn't support negatives)
	"SPIDER.mmvalid":5, # 0 if invalid, 1 if next 4 values are valid (unused in EMAN3), in the 70s it was expensive to compute these
	"SPIDER.max":6, 
	"SPIDER.min":7, 
	"SPIDER.mean":8,
	"SPIDER.sigma":9, 
	"nx":11,
	"SPIDER.headrec":12,	# number of records in header, record is nx*4
	"SPIDER.angvalid":13,	# 1 if next 7(?) values are valid
	"SPIDER.phi":14,
	"SPIDER.theta":15,
	"SPIDER.gamma":16,
	"SPIDER.dx":17,
	"SPIDER.dy":18,
	"SPIDER.dz":19,
	"SPIDER.scale":20,
	"SPIDER.headlen":21,	# header length in bytes
	"SPIDER.reclen":22,		# record length in bytes (header is integer multiple of reclen)
	"SPIDER.istack":23,		# 0 for single image file, 2 for stack file, -1 for individual image headers
							# if negative in overall header, indicates "indexed" stack, which we do not support
	"SPIDER.maxim":25,		# in overall stack file header, number of highest numbered image in stack, first image 1
	"SPIDER.imgnum":26,		# number of the current image in the stack, 0 if image empty
	"SPIDER.lastidx":27		# in overall header, highest index in use (unused in EMAN3)
 }

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HeaderInvalid(Exception):
	"""Exception for invalid header metadata"""
	pass


# ---------------------------------------------------------------------------
# SpiderIO class
# ---------------------------------------------------------------------------

class SpiderIO:
	"""Read and write SPIDER format image files.

	SPIDER files store float32 image data with rich binary headers.
	Supported flavors: single 2D/3D images and homogeneous stacks.

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with SpiderIO("myfile.spi", "r") as spider:
			meta = spider.read_header(0)	  # dict with all header fields
			data = spider.read_data(0)		  # numpy array (nz, ny, nx)

		with SpiderIO("out.spi", "w") as spider:
			spider.write_header({"nx":128,"ny":128,"nz":1})
			spider.write_data(np.random.rand(1, 128, 128).astype(np.float32))

	Attributes (read-only after init):
		filename	 - path to the SPIDER file.
		is_stack	 - True if this is a stack file, False for single image.
		nimg		 - total number of images in the file.
		nx, ny, nz	 - dimensions from the overall header.
		big_endian	 - endianness of the on-disk data.
	"""

	# Class capability flags
	SUPPORT_STACK = True
	SUPPORT_3D = True
	SUPPORT_3D_STACK = False
	SUPPORT_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode
		self._file = None
		self._is_new_file = not os.path.exists(self.filename)
		self._initialized = False

		# Convenience properties set during init
		self.is_stack = None
		self.nimg = None
		self.nx = None
		self.ny = None
		self.nz = None
		self.big_endian = False		# whether the file on disk is big endian
		self.reclen = None
		self.headlen = 1024			# initial value, gets corrected when file header is read

		if self.mode=="rw":
			# For read-write, create if it doesn't exist
			if self._is_new_file: fmode="wb+"
			else: 
				fmode="rb+"		# "wb+" would truncate!
				self._initialized=True
		elif self.mode=="r": 
			fmode="rb"
			self._initialized=True
		else: raise SpiderIOError(f"Invalid mode '{self.mode}'. Use 'r' or 'rw'.")

		try:
			self._file = open(self.filename, fmode)
		except OSError as e:
			raise SpiderIOError(f"Cannot open file '{self.filename}': {e}")

		# This will populate information about the file. Otherwise it happens with the 
		# first write_header call
		if self._initialized : self.read_header(-1)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None: self._file.close()
		return False

	def __len__(self):
		"""Return number of images in the file."""
		return self.nimg if self.nimg else 0

	# def __del__(self):
	# 	try:
	# 		self._file.flush()
	# 	except Exception:
	# 		pass
		

	# ---------------------------------------------------------------
	# Validation
	# ---------------------------------------------------------------

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. 0 = float32."""
		return [0]

	@staticmethod
	def is_valid(path,chunk):
		"""Check whether a file is potentially a valid SPIDER image without opening.

		takes a filename (for extension checking) and the first 1k of the 
		file as a binary string (read once by the calling function for
		checking multiple formats)
		"""

		if struct.unpack(">f",chunk[16:20])[0] in (1.0,3.0,-11.0,-12.0,-21.0,-22.0) : big_endian = True
		elif struct.unpack("<f",chunk[16:20])[0] in (1.0,3.0,-11.0,-12.0,-21.0,-22.0) : big_endian = False
		else: return False

		if big_endian: floats = np.frombuffer(chunk[:SPIDER_FLOATS_IN_HEADER*4], dtype='>f')
		else: floats = np.frombuffer(chunk[:SPIDER_FLOATS_IN_HEADER*4], dtype='<f')
		
		# not exhaustive, but should be a pretty strong check
		if (floats[_HEADER_FLOATS["SPIDER.maxim"]] != int(floats[_HEADER_FLOATS["SPIDER.maxim"]]) or
			floats[_HEADER_FLOATS["nx"]] != int(floats[_HEADER_FLOATS["nx"]]) or
			floats[_HEADER_FLOATS["ny"]] != int(floats[_HEADER_FLOATS["ny"]]) or
			floats[_HEADER_FLOATS["nz"]] != int(floats[_HEADER_FLOATS["nz"]]) or
			floats[_HEADER_FLOATS["nx"]] < 1) : return False

		return True

	# ---------------------------------------------------------------
	# Read operations
	# ---------------------------------------------------------------
	def read_header(self, index):
		"""Read the SPIDER header for one image and return it as a dict.

		If index == -1, reads the overall stack header instead of
		a per-image header.	 Returns a Python dict with all SPIDER fields
		and additional keys like "nx", "ny", "nz", "datatype".

		Parameters:
			index : int  Image number (0-biased). Use -1 for overall header.

		Returns:
			dict containing header metadata.
		"""
		
		if index==0 and self._initialized and not self.is_stack : index=-1		# single image file only has the global header

		if index==-1: 
			self._file.seek(0)
			raw = self._file.read(SPIDER_HEADER_SIZE)
			if len(raw) != SPIDER_HEADER_SIZE:
				raise IOError("Incomplete SPIDER header read")

			if struct.unpack(">f",raw[16:20])[0] in (1.0,3.0,-11.0,-12.0,-21.0,-22.0) : self.big_endian = True
			elif struct.unpack("<f",raw[16:20])[0] in (1.0,3.0,-11.0,-12.0,-21.0,-22.0) : self.big_endian = False
			else: raise HeaderInvalid("Invalid Spider file type in header")
		else:
			self._file.seek(self.headlen+(self.headlen+self.reclen*self.ny*self.nz)*index)	# seek to the correct header
			raw=self._file.read(SPIDER_HEADER_SIZE)
			if len(raw) != SPIDER_HEADER_SIZE:
				raise IOError("Incomplete SPIDER header read")

		# First 844 bytes: treat as float array and swap if needed (matching C++)
		if self.big_endian: floats = np.frombuffer(raw[:SPIDER_FLOATS_IN_HEADER*4], dtype='>f')
		else: floats = np.frombuffer(raw[:SPIDER_FLOATS_IN_HEADER*4], dtype='<f')
	
		# Char fields at offsets 844, 855, 863 within the raw struct bytes.
		date_str = raw[844:855].decode('ascii', errors='replace').lstrip('\x00')[:11]
		time_str = raw[855:863].decode('ascii', errors='replace').lstrip('\x00')[:8]
		title_str = raw[863:].decode('ascii', errors='replace').lstrip('\x00')
	
		header = {k:floats[v] for k,v in _HEADER_FLOATS.items()}
		header["SPIDER_date"]=date_str
		header["SPIDER_time"]=time_str
		header["SPIDER_title"]=title_str
		header["bitdepth"] = 0
		
		if index==-1:
			self.nimg=int(header["SPIDER.maxim"]) # SPIDER numbers are EMAN numbers+1
			self.nx=int(header["nx"])
			self.ny=int(header["ny"])
			self.nz=int(header["nz"])
			self.is_stack = header["SPIDER.istack"]==2
			self.reclen=self.nx*4
			self.headlen=ceil(1024/self.reclen)*self.reclen

		return header

	def read_data(self, index=0):
		"""Read the float32 image data for one SPIDER image.

		Returns a numpy array of shape (nz, ny, nx). The on-disk format
		stores data in z-fastest (Fortran-style) order.

		Parameters:
			index : int  Image number (0-based).

		Returns:
			numpy.ndarray[float32] shaped (nz, ny, nx).
		"""
		if index < 0 or index >= self.nimg:
			raise SpiderReadError(
				f"Image index {index} out of range [0, {self.nimg})")

		self._file.seek(self.headlen*2+(self.headlen+self.reclen*self.ny*self.nz)*index)

		nbytes = self.nx * self.ny * self.nz * 4
		raw = self._file.read(nbytes)
		if len(raw) < nbytes:
			raise FileIOError(f"Incomplete data read for image {index}: (got {len(raw)} of {nbytes} bytes)")

		if self.big_endian: data = np.frombuffer(raw, dtype='>f')
		else: data = np.frombuffer(raw, dtype='<f')

		# Reshape in Fortran order to get (nz, ny, nx) with z-fastest
		if self.nz==1: return data.reshape((self.nx, self.ny), order='F').transpose(1, 0)
		else: return data.reshape((self.nx, self.ny, self.nz), order='F').transpose(2, 1, 0)

	# ---------------------------------------------------------------
	# Write operations
	# ---------------------------------------------------------------

	def write_header(self, meta, index=-1):
		"""Write a SPIDER header (overall or per-image).

		meta must at minimum contain "nx", "ny", and "nz".
		Optional keys: "minimum", "maximum", "mean", "sigma",
		"SPIDER.title", "xform.projection" / "xform.align3d" dicts.

		If index == -1 (default), appends a new image to the stack.

		Returns the image index that was written.
		"""
		
		if self.mode=="r" : raise FileIOError("SPIDER file opened read-only, writes forbidden")

		# Validate bitdepth (SPIDER is float32 only)
		bitdepth = int(meta.get("bitdepth", 0))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"SPIDER only supports float32 (bitdepth=0); got {bitdepth}")

		# if this is our first write on a new file, we initialize ourselves from the first header
		if self.nx is None: self.nx=int(meta["nx"])
		if self.ny is None: self.ny=int(meta["ny"])
		if self.nz is None: self.nz=int(meta["nz"])
		if self.reclen is None:
			self.reclen=self.nx*4
			self.headlen=ceil(1024/self.reclen)*self.reclen

		# Resolve index: -1 means append. First image goes at 0.
		if index < 0:
			if self.nimg is None or self.nimg == 0:
				index = 0
			else:
				index = self.nimg
		self.nimg = max(self.nimg or 0, index + 1)

		# we force a few specific values, modification of meta is an intentional side-effect
		meta["SPIDER.istack"]=2
		meta["nx"]=self.nx
		meta["ny"]=self.ny
		meta["nz"]=self.nz
		meta["SPIDER.type"]=1 if self.nz==1 else 3
		meta["SPIDER.reclen"]=self.reclen
		meta["SPIDER.headrec"]=int(self.headlen/self.reclen)
		meta["SPIDER.headlen"]=self.headlen
		meta["SPIDER.istack"]=0		# this will be 2 in the global header (we only write stacks)
		meta["SPIDER.imgnum"]=index+1	# should be 0 in the global header
		meta["SPIDER.maxim"]=0		# this will get reset in the global header after each write, but should be zero for individual images

		# we use 256 here so tobytes())gives us the full 1k header
		if self.big_endian: floats=np.zeros(256,dtype='>f')
		else: floats=np.zeros(256,dtype="<f")

		# all unassigned values are set to 0
		for k,v in _HEADER_FLOATS.items(): 
			try: floats[v]=meta[k]
			except: pass			# ignore any missing metadata keys

		# raw is the header represented as bytes
		raw=bytearray(floats.tobytes())
		raw[1023]=0

		from datetime import datetime
		now = datetime.now()
		sp_date = meta.get("SPIDER_date", now.strftime("%d-%b-%Y"))[:11]
		sp_time = meta.get("SPIDER_time", now.strftime("%H:%M:%S"))[:8]
		sp_title = meta.get("SPIDER_title", "EMAN3")[0:160]

		raw[844:855]=sp_date.encode('ascii', errors='ignore')
		raw[855:863]=sp_time.encode('ascii', errors='ignore')
		raw[863:1023]=sp_title.encode('ascii', errors='ignore').ljust(160, b'\x00')[:160]

		# write the same header we're writing for the image, then fix it at the end
		if not self._initialized :
			self._file.seek(0)
			self._file.write(raw)

			self._file.seek(_HEADER_FLOATS["SPIDER.istack"]*4)
			self._file.write(struct.pack(">f", 2.0) if self.big_endian else struct.pack("<f", 2.0))

			self._file.seek(_HEADER_FLOATS["SPIDER.imgnum"]*4)
			self._file.write(b"\x00\x00\x00\x00")		# 0.0
			# maxim updated below

		# write the individual header
		if index<0: index=self.nimg
		self._file.seek(self.headlen+(self.headlen+self.reclen*self.ny*self.nz)*index)
		self._file.write(raw)

		# Update overall header if we're growing the stack
		if index >= self.nimg:
			self.nimg = index + 1
		
		# Rewrite image count
		self._file.seek(_HEADER_FLOATS["SPIDER.maxim"]*4)
		if self.big_endian : self._file.write(struct.pack(">f",self.nimg))
		else : self._file.write(struct.pack("<f",self.nimg))

		return index

	def write_data(self, data, index):
		"""Write float32 image data at the given index.

		data should be numpy array of shape (nz, ny, nx). Header must
		have been written first (via write_header). index cannot be provided
		as -1, should use the return value of write_header
		"""

		if index<0: raise ValueError("index must be >=0, use return value from write_header")

		if self.nz==1:
			if self.big_endian: write_data = data.astype(">f").ravel().tobytes()
			else: write_data = data.astype("<f").ravel().tobytes()			
		else:
			if self.big_endian: write_data = data.astype(">f").ravel().tobytes()
			else: write_data = data.astype("<f").ravel().tobytes()

		if len(write_data)!=self.reclen*self.ny*self.nz: raise InvalidDimensions(f"Data size {data.shape} != expected ({self.nz},{self.ny},{self.nx})")

		self._file.seek(self.headlen*2+(self.headlen+self.reclen*self.ny*self.nz)*index)
		self._file.write(write_data)

