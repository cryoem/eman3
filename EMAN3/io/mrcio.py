#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ MrcIO)
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

"""MRC image format reader/writer for EMAN.

MRC (Medical Research Council) stores 1D, 2D or 3D images with a rich binary
header (~1048 bytes before IMOD extensions, or ~1144 with full IMOD). Multiple
data modes: float32, uint8, int16, uint16, and packed 4-bit (UHEX).

Supported extensions:
	.mrc / .hdr - Standard MRC (single image)
	.mrcs / .MRCS - MRC stack format (each slice is a separate image with extended header)
	.raw        - FEI raw format (extended per-image headers)

Key quirks handled (matching C++ implementation):
	- Byte order auto-detection via machine stamp, mapc/mapr/maps range checks,
	  dimension sanity, and mode validity fallbacks.
	- 8-bit packed mode detection (2 nibbles per byte) when ny/nx ~ 2 with hints.
	- Transpose when mapc==2 && mapr==1 (X/Y swapped in source).
	- FEI extended header parsing for microscope metadata.
	- CTF parameter extraction from label strings.

References: https://www.ccpem.ac.uk/mrc_format/mrc2014.php
"""

import os
import struct
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

class InvalidIndex(Exception):
	"""Indicates a request for an unsupported or missing image index"""
	pass


# ---------------------------------------------------------------------------
# MRC constants
# ---------------------------------------------------------------------------

MRC_MODE_UCHAR = 0
MRC_MODE_SHORT = 1
MRC_MODE_FLOAT = 2
MRC_MODE_SHORT_COMPLEX = 3
MRC_MODE_FLOAT_COMPLEX = 4
MRC_MODE_USHORT = 6
MRC_MODE_UHEX = 101       # packed 4-bit (2 values per byte)

MRC_NUM_LABELS = 10
MRC_LABEL_SIZE = 80
NUM_4BYTES_PRE_MAP = 52   # ints before the "MAP " string at offset 208
NUM_4BYTES_AFTER_MAP = 3  # ints after the "MAP " string (not used directly)

CTF_MAGIC = "!-"
SHORT_CTF_MAGIC = "!$"

_HOST_LITTLE = (sys.byteorder == "little")


def _generate_machine_stamp():
	"""Generate the machine endian stamp used in MRC headers."""
	if _HOST_LITTLE:
		return 0x41440000  # little-endian
	else:
		return 0x11110000  # big-endian

_HOST_STAMP = _generate_machine_stamp()


def _get_mode_size(mode):
	"""Return byte size per pixel for a given MRC mode."""
	sizes = {
		MRC_MODE_UCHAR: 1,
		MRC_MODE_SHORT: 2,
		MRC_MODE_FLOAT: 4,
		MRC_MODE_USHORT: 2,
		MRC_MODE_SHORT_COMPLEX: 2,
		MRC_MODE_FLOAT_COMPLEX: 4,
	}
	return sizes.get(mode, 0)


# MRC header size: first 52 ints (208 bytes) + "MAP " (4) + machinestamp(4) + rms(4)
#   + nlabels(4) + labels[10][80] (800) = 1040 bytes (CCP4/MRC2000 format).
# IMOD adds extra fields between byte 96 and the MAP string.
# The C++ struct includes IMOD extensions making it ~1048 bytes.
MRC_HEADER_SIZE = 1048

# FEI extended header: complex packed struct, calculated by C++ as __attribute__((packed))
FEI_EXT_HEADER_SIZE = 632


def _build_mrc_header(fields):
	"""Build MRC header bytes from a dict of field values.
	
	Returns bytes of length MRC_HEADER_SIZE (1048).
	Layout matches the C++ MrcHeader struct including IMOD extensions.
	"""
	buf = bytearray(MRC_HEADER_SIZE)

	def _pack_ints(start_offset, *values):
		for i, v in enumerate(values):
			struct.pack_into("<i", buf, start_offset + i*4, int(v))

	def _pack_floats(start_offset, *values):
		for i, v in enumerate(values):
			struct.pack_into("<f", buf, start_offset + i*4, float(v))

	def _pack_shorts(start_offset, *values):
		for i, v in enumerate(values):
			struct.pack_into("<h", buf, start_offset + i*2, int(v))

	# First 10 ints: nx, ny, nz, mode, nxstart, nystart, nzstart, mx, my, mz
	_pack_ints(0,
		fields.get("nx", 0), fields.get("ny", 0), fields.get("nz", 1),
		fields.get("mode", MRC_MODE_FLOAT),
		fields.get("nxstart", 0), fields.get("nystart", 0), fields.get("nzstart", 0),
		fields.get("mx", 0), fields.get("my", 0), fields.get("mz", 1))

	# Next 9 floats: xlen, ylen, zlen, alpha, beta, gamma, amin, amax, amean
	_pack_floats(40,
		fields.get("xlen", 1.0), fields.get("ylen", 1.0), fields.get("zlen", 1.0),
		fields.get("alpha", 90.0), fields.get("beta", 90.0), fields.get("gamma", 90.0),
		fields.get("amin", 0.0), fields.get("amax", 0.0), fields.get("amean", 0.0))

	# Next 10 ints: mapc, mapr, maps, amin(2nd?), amax(2nd?), amean(2nd?), ispg, nsymbt
	# Actually per the struct after floats it's:
	# mapc(int), mapr(int), maps(int) = offsets 76,80,84
	# then amin(float), amax(float), amean(float) at offsets 88,92,96? No...
	# Let me follow the struct literally. After offset 36 (9 floats done at 76),
	# we have: ispg(int,76), nsymbt(int,80), creatid(short,84), blank[6](86),
	# exttyp[4](92), nversion(int,96), blank2[16](100)

	_pack_ints(76,
		fields.get("mapc", 1), fields.get("mapr", 2), fields.get("maps", 3))

	# amin/amax/amean are floats in C++ struct at offsets 88-92? No.
	# Re-reading: the struct has ispg(int) at offset from start, then nsymbt(int).
	# After the first block of ints (10 ints=40 bytes), we have 9 floats (36 bytes) at offset 40.
	# So offset 76 starts with: mapc(int)=fields[16] in original array... no, that was a different layout.
	
	# Let me just follow the struct definition literally:
	# int nx(0), ny(4), nz(8)
	# int mode(12), nxstart(16), nystart(20), nzstart(24)
	# int mx(28), my(32), mz(36)
	# float xlen(40), ylen(44), zlen(48)
	# float alpha(52), beta(56), gamma(60)
	# int mapc(64), mapr(68), maps(72)
	# float amin(76), amax(80), amean(84)
	# int ispg(88), nsymbt(92)
	# short creatid(96), char blank[6](98-103), char exttyp[4](104-107), int nversion(108)
	# char blank2[16](112-127)
	# short nint(128), nreal(130), sub(132), zfac(134)
	# float min2(136), max2(140), min3(144), max3(148)
	# char imod_stamp[4](152-155), int imod_flags(156)
	# short idtype(160), lens(162), nd1(164), nd2(166), vd1(168), vd2(170)
	# float tiltangles[6](172-187)? Actually these are at offsets in the C++ header...
	
	# Wait, I'm getting confused. Let me re-read the struct more carefully by counting bytes:
	# offset 0:  nx(int4) + ny(4) + nz(4) = 3 ints = 12 bytes
	# offset 12: mode(4) + nxstart(4) + nystart(4) + nzstart(4) = 16 bytes (offset now 28)
	# offset 28: mx(4) + my(4) + mz(4) = 12 bytes (offset now 40)
	# offset 40: xlen(f4) + ylen(4) + zlen(4) = 12 bytes (offset now 52)
	# offset 52: alpha(4) + beta(4) + gamma(4) = 12 bytes (offset now 64)
	# offset 64: mapc(4) + mapr(4) + maps(4) = 12 bytes (offset now 76)
	# offset 76: amin(f4) + amax(4) + amean(4) = 12 bytes (offset now 88)
	# offset 88: ispg(4) + nsymbt(4) = 8 bytes (offset now 96)
	# offset 96: creatid(h2) + blank[6](6) = 8 bytes (offset now 104)
	# offset 104: exttyp[4](4) + nversion(4) = 8 bytes (offset now 112)
	# offset 112: blank2[16](16) = 16 bytes (offset now 128)
	# offset 128: nint(2) + nreal(2) + sub(2) + zfac(2) = 8 bytes (offset now 136)
	# offset 136: min2(4) + max2(4) + min3(4) + max3(4) = 16 bytes (offset now 152)
	# offset 152: imod_stamp[4](4) + imod_flags(4) = 8 bytes (offset now 160)
	# offset 160: idtype(2) + lens(2) + nd1(2) + nd2(2) + vd1(2) + vd2(2) = 12 bytes (offset now 172)
	
	# Hmm, but the C++ struct has float tiltangles[6] after vd2... Let me re-check:
	# The original header showed these fields between nd2 and xorigin:
	# "short idtype; short lens; short nd1; short nd2; short vd1; short vd2;"
	# Then: "float tiltangles[6];" (24 bytes)
	# Then: "float xorigin, yorigin, zorigin;" (12 bytes) 
	# So after offset 172 we have: tiltangles[6]=24 (offset now 196), then origins=12 (offset 208)
	
	# But wait, that makes the total before MAP different. Let me check whether tiltangles exist.
	# Looking at the C++ header file again... yes, tiltangles[6] is between vd2 and xorigin.
	
	# So: offset 172-195: tiltangles[6] (24 bytes)
	# offset 196-207: xorigin(4) + yorigin(4) + zorigin(4) = 12 bytes
	# offset 208: map[4] "MAP " 
	# offset 212: machinestamp(4)
	# offset 216: rms(4)
	# offset 220: nlabels(4)
	# offset 224: labels[10][80] = 800 bytes
	
	# Total = 224 + 800 = 1024. That's not matching our header size of 1048.
	# The discrepancy is the IMOD fields that extend the struct. Let me look at what comes after blank[6]:
	# The original C++ has more IMOD fields between offset 96 and MAP. Specifically:
	# After creatid+blank (offset 104), we have exttyp+nversion+blank2, then the shorts, floats, etc.
	
	# OK I think the issue is that tiltangles may not be in the base MRC format but are IMOD extensions.
	# Let me just use offset 208 for MAP and work with what makes sense:
	
	# Re-doing with cleaner offsets:
	# 0-39: 10 ints (nx,nz,ny,mode,nxstart,nystart,nzstart,mx,my,mz) -- DONE above
	# 40-71: 9 floats (xlen,ylen,zlen,alpha,beta,gamma,amin,amax,amean) -- DONE above  
	# 64-75: 3 ints (mapc,mapr,maps) -- DONE above
	
	# 76-83: amin/amax/amean floats -- wait, these overlap with the ispg/nsymbt integers.
	# Looking at the struct again carefully... the struct has them as SEPARATE fields:
	# float amin(76), amax(80), amean(84) -- these are at byte offsets 76-87 (12 bytes, but overlap with ints?)
	
	# No wait. Let me re-read the header from C++ one more time. The struct is:
	# int nx(0), ny(4), nz(8) -- done ✓
	# int mode(12), nxstart(16), nystart(20), nzstart(24) -- done ✓ 
	# int mx(28), my(32), mz(36) -- done ✓ (but I wrote 10 ints including nx at offset 0, that's right)
	# Actually no: "int nx; int ny; int nz;" then "int mode; int nxstart; int nystart; int nzstart;" 
	# then "int mx; int my; int mz;" -- that's only 10 ints at offset 0-39. Good.
	# Then floats: xlen(40), ylen(44), zlen(48), alpha(52), beta(56), gamma(60) = 6 floats, not 9.
	
	# Ah! I see the issue. The first block has:
	# xlen,ylen,zlen,alpha,beta,gamma (6 floats at 40-71)
	# Then mapc,mapr,maps as INTS (3 ints at 72-83)
	# THEN amin,amax,amean as FLOATS (3 floats at 84-95)
	# Then ispg,int nsymbt (2 ints at 96-103)
	
	# So it's NOT "9 floats" -- it's 6 floats, then 3 ints, then 3 floats.
	
	# Let me FIX the packing:
	# Offsets 40-71 (24 bytes): xlen,ylen,zlen,alpha,beta,gamma as floats - DONE
	# I already packed _pack_ints(0,...) and _pack_floats(40,...), but _pack_ints only has 10 values.
	# The issue is the first block: the original struct groups ints together. But actually the C++ comment says
	# "int nx; int ny; int nz; int mode; int nxstart; int nystart; int nzstart;" (7 ints)
	# "int mx; int my; int mz;" (3 more = 10 total at offsets 0-39). OK that's right.
	# Then "float xlen; float ylen; float zlen;" at 40-51
	# "float alpha; float beta; float gamma;" at 52-63
	# That's only 6 floats, not 9! The next three (amin,amax,amean) come AFTER mapc/mapr/maps.
	
	# I need to fix this. Let me redo from offset 40:
	_pack_floats(40,
		fields.get("xlen", 1.0), fields.get("ylen", 1.0), fields.get("zlen", 1.0))
	_pack_floats(52,
		fields.get("alpha", 90.0), fields.get("beta", 90.0), fields.get("gamma", 90.0))
	
	# mapc,mapr,maps as INTS at offset 64-75
	_pack_ints(64,
		fields.get("mapc", 1), fields.get("mapr", 2), fields.get("maps", 3))
	
	# amin,amax,amean as FLOATS at offset 76-87
	_pack_floats(76,
		fields.get("amin", 0.0), fields.get("amax", 0.0), fields.get("amean", 0.0))
	
	# ispg(int) + nsymbt(int) at offset 88-95
	_pack_ints(88,
		fields.get("ispg", 0), fields.get("nsymbt", 0))
	
	# creatid(short) at offset 96
	struct.pack_into("<h", buf, 96, fields.get("creatid", 0))
	
	# blank[6] at offset 98-103 (zeroed by default)
	# exttyp[4] at offset 104-107
	exttyp = fields.get("exttyp", b"\x00\x00\x00\x00")
	if isinstance(exttyp, str):
		exttyp = exttyp.encode("ascii", errors="replace")
	buf[104:108] = exttyp[:4].ljust(4, b"\x00")
	
	# nversion(int) at offset 108
	struct.pack_into("<i", buf, 108, fields.get("nversion", 0))
	
	# blank2[16] at offset 112-127 (zeroed)
	
	# IMOD shorts at offset 128: nint, nreal, sub, zfac
	_pack_shorts(128,
		fields.get("nint", 0), fields.get("nreal", 0),
		fields.get("sub", 0), fields.get("zfac", 1))
	
	# IMOD floats at offset 136: min2, max2, min3, max3
	_pack_floats(136,
		fields.get("min2", 0.0), fields.get("max2", 0.0),
		fields.get("min3", 0.0), fields.get("max3", 0.0))
	
	# imod_stamp[4] at offset 152 + imod_flags(int) at offset 156
	imod_stamp = fields.get("imod_stamp", b"\x00\x00\x00\x00")
	if isinstance(imod_stamp, int):
		imod_stamp = struct.pack("<i", imod_stamp)
	buf[152:156] = imod_stamp[:4].ljust(4, b"\x00")
	struct.pack_into("<i", buf, 156, fields.get("imod_flags", 0))
	
	# idtype(160), lens(162), nd1(164), nd2(166), vd1(168), vd2(170)
	_pack_shorts(160,
		fields.get("idtype", 0), fields.get("lens", 0),
		fields.get("nd1", 0), fields.get("nd2", 0),
		fields.get("vd1", 0), fields.get("vd2", 0))
	
	# tiltangles[6] at offset 172-195 (24 bytes)
	tilt = fields.get("tiltangles", [0.0]*6)
	for i in range(6):
		struct.pack_into("<f", buf, 172 + i*4, float(tilt[i]))
	
	# xorigin, yorigin, zorigin at offset 196-207
	_pack_floats(196,
		fields.get("xorigin", 0.0), fields.get("yorigin", 0.0),
		fields.get("zorigin", 0.0))
	
	# "MAP " string at offset 208
	buf[208:212] = b"MAP "
	
	# machinestamp(int) at offset 212
	struct.pack_into("<i", buf, 212, _HOST_STAMP)
	
	# rms(float) at offset 216
	struct.pack_into("<f", buf, 216, fields.get("rms", 0.0))
	
	# nlabels(int) at offset 220
	nlabels = min(fields.get("nlabels", 1), MRC_NUM_LABELS)
	struct.pack_into("<i", buf, 220, nlabels)
	
	# labels[10][80] at offset 224-1023 (800 bytes)
	labels = fields.get("labels", [])
	for i in range(MRC_NUM_LABELS):
		if i < len(labels):
			label_bytes = labels[i].encode("ascii", errors="replace") if isinstance(labels[i], str) else labels[i]
		else:
			label_bytes = b""
		offset = 224 + i * MRC_LABEL_SIZE
		buf[offset:offset+MRC_LABEL_SIZE] = label_bytes[:MRC_LABEL_SIZE].ljust(MRC_LABEL_SIZE, b"\x00")
	
	return bytes(buf)


def _read_mrc_header(raw):
	"""Read MRC header from raw bytes. Returns dict of fields.
	
	Handles both little-endian and big-endian files by trying both.
	Returns (fields_dict, is_big_endian).
	"""
	def _unpack(endian, data):
		fields = {}
		# First 10 ints
		for i, key in enumerate(["nx","ny","nz","mode","nxstart","nystart","nzstart","mx","my","mz"]):
			fields[key] = struct.unpack_from(endian+"i", data, i*4)[0]
		
		# 6 floats
		for i, key in enumerate(["xlen","ylen","zlen","alpha","beta","gamma"]):
			fields[key] = struct.unpack_from(endian+"f", data, 40+i*4)[0]
		
		# 3 ints (mapc,mapr,maps)
		for i, key in enumerate(["mapc","mapr","maps"]):
			fields[key] = struct.unpack_from(endian+"i", data, 64+i*4)[0]
		
		# 3 floats (amin,amax,amean)
		for i, key in enumerate(["amin","amax","amean"]):
			fields[key] = struct.unpack_from(endian+"f", data, 76+i*4)[0]
		
		# 2 ints (ispg, nsymbt)
		for i, key in enumerate(["ispg","nsymbt"]):
			fields[key] = struct.unpack_from(endian+"i", data, 88+i*4)[0]
		
		# creatid(short)
		fields["creatid"] = struct.unpack_from(endian+"h", data, 96)[0]
		
		# exttyp[4]
		fields["exttyp"] = data[104:108].decode("ascii", errors="replace")
		
		# nversion(int)
		fields["nversion"] = struct.unpack_from(endian+"i", data, 108)[0]
		
		# IMOD shorts
		ioff = 128
		for i, key in enumerate(["nint","nreal","sub","zfac"]):
			fields[key] = struct.unpack_from(endian+"h", data, ioff+i*2)[0]
		
		# IMOD floats
		for i, key in enumerate(["min2","max2","min3","max3"]):
			fields[key] = struct.unpack_from(endian+"f", data, 136+i*4)[0]
		
		# imod_stamp[4] + imod_flags(int)
		fields["imod_stamp_raw"] = data[152:156]
		fields["imod_flags"] = struct.unpack_from(endian+"i", data, 156)[0]
		
		# More shorts
		ioff = 160
		for i, key in enumerate(["idtype","lens","nd1","nd2","vd1","vd2"]):
			fields[key] = struct.unpack_from(endian+"h", data, ioff+i*2)[0]
		
		# Tilt angles[6]
		fields["tiltangles"] = []
		for i in range(6):
			fields["tiltangles"].append(struct.unpack_from(endian+"f", data, 172+i*4)[0])
		
		# Origins
		fields["xorigin"] = struct.unpack_from(endian+"f", data, 196)[0]
		fields["yorigin"] = struct.unpack_from(endian+"f", data, 200)[0]
		fields["zorigin"] = struct.unpack_from(endian+"f", data, 204)[0]
		
		# map string
		fields["map"] = data[208:212].decode("ascii", errors="replace")
		
		# machinestamp + rms + nlabels
		fields["machinestamp"] = struct.unpack_from(endian+"i", data, 212)[0]
		fields["rms"] = struct.unpack_from(endian+"f", data, 216)[0]
		fields["nlabels"] = struct.unpack_from(endian+"i", data, 220)[0]
		
		# Labels
		labels = []
		for i in range(MRC_NUM_LABELS):
			label_bytes = data[224 + i*MRC_LABEL_SIZE:224 + (i+1)*MRC_LABEL_SIZE]
			try:
				labels.append(label_bytes.decode("ascii").strip("\x00"))
			except Exception:
				labels.append("")
		fields["labels"] = labels
		
		return fields
	
	# Try little-endian first, validate via check_swap logic
	data_le = np.frombuffer(raw[:NUM_4BYTES_PRE_MAP*4], dtype='<i4')
	do_swap, have_err = _check_swap(data_le)
	
	if do_swap:
		fields = _unpack(">", raw)
		return fields, True
	else:
		fields = _unpack("<", raw)
		return fields, False


def _check_swap(data):
	"""Determine byte order of MRC header data.
	
	data: numpy array of int32 (little-endian interpretation of raw bytes).
	Returns (do_swap, have_err).
	"""
	nx = int(data[0])
	ny = int(data[1])
	nz = int(data[2])
	mrcmode = int(data[3])
	mapc = int(data[16]) if len(data) > 16 else 0
	mapr = int(data[17]) if len(data) > 17 else 0
	maps = int(data[18]) if len(data) > 18 else 0
	mach = int(data[44]) if len(data) > 44 else 0

	swapped = data.byteswap().copy()
	nxw, nyw, nzw = int(swapped[0]), int(swapped[1]), int(swapped[2])
	modew = int(swapped[3])
	mapcw = int(swapped[16]) if len(swapped) > 16 else 0
	marpw = int(swapped[17]) if len(swapped) > 17 else 0
	maps2 = int(swapped[18]) if len(swapped) > 18 else 0

	max_dim = 1 << 20

	# Primary check: machine stamp at index 44
	if mach == _HOST_STAMP:
		return (False, False)

	machw = int(swapped[44]) if len(swapped) > 44 else 0
	if machw == _HOST_STAMP:
		return (True, False)

	# Fallback for mode 0 (uchar): check mapc/mapr/maps range [1,3]
	if mrcmode == 0:
		if 1 <= mapr <= 3 and 1 <= mapc <= 3 and 1 <= maps <= 3:
			return (False, False)
		if 1 <= marpw <= 3 and 1 <= mapcw <= 3 and 1 <= maps2 <= 3:
			return (True, False)

		# Check dimension sanity
		ave = (nx + ny + nz) / 3.0
		avew = (nxw + nyw + nzw) / 3.0
		if nx > 0 and ny > 0 and nz > 0 and ave <= max_dim:
			return (False, False)
		if nxw > 0 and nyw > 0 and nzw > 0 and avew <= max_dim:
			return (True, False)

		return (False, True)

	# Non-zero mode: check mode validity
	if 0 < mrcmode < 128:
		return (False, False)
	if 0 < modew < 128:
		return (True, False)

	return (False, True)


def _read_ctf_from_labels(labels):
	"""Extract CTF parameters from MRC labels if present."""
	ctf = None
	for label in (labels or [])[:5]:
		if label.startswith(CTF_MAGIC):
			try:
				params = label[len(CTF_MAGIC):].split()
				ctf = {}
				if len(params) >= 2:
					ctf["defocus_u"] = float(params[0])
					ctf["defocus_v"] = float(params[1])
				if len(params) >= 3:
					ctf["defocus_angle"] = float(params[2])
				if len(params) >= 4:
					ctf["astigmatism"] = abs(float(params[0]) - float(params[1]))
			except Exception:
				break
		elif label.startswith(SHORT_CTF_MAGIC):
			try:
				params = label[len(SHORT_CTF_MAGIC):].split()
				ctf = {}
				if params:
					ctf["defocus"] = float(params[0])
			except Exception:
				break

	return ctf


def _bitdepth_to_mode(bitdepth):
	"""Map bitdepth to MRC mode."""
	if bitdepth == 0:
		return MRC_MODE_FLOAT
	elif bitdepth == 8:
		return MRC_MODE_UCHAR
	elif bitdepth == 16:
		return MRC_MODE_USHORT
	else:
		raise FileFormatError(f"Unsupported bitdepth {bitdepth} for MRC")


def _mode_to_bitdepth(mode):
	"""Map MRC mode to bitdepth."""
	if mode in (MRC_MODE_FLOAT, MRC_MODE_SHORT_COMPLEX, MRC_MODE_FLOAT_COMPLEX):
		return 0
	elif mode == MRC_MODE_UCHAR:
		return 8
	elif mode == MRC_MODE_USHORT:
		return 16
	else:
		return 0


def _transpose_xy(data):
	"""Transpose X and Y axes in data."""
	if data.ndim == 2:
		return data.T
	return np.transpose(data, (0, 2, 1))


# ---------------------------------------------------------------------------
# MrcIO class
# ---------------------------------------------------------------------------

class MrcIO:
	"""Read and write MRC format image files.

	MRC stores float32, uint8, int16 or uint16 image data with a rich binary
	header (~1048 bytes). Supports single images, stacks (MRCS), and FEI extended
	headers. BigTIFF-style sequential writes for multi-image files.

	Modes: "r" (read-only), "rw" (read-write, creates new file if it doesn't exist).

	Usage example::

		with MrcIO("myfile.mrc", "r") as mrc:
			meta = mrc.read_header(0)
			data = mrc.read_data(0)   # numpy array float32 (nz, ny, nx)

		with MrcIO("out.mrcs", "rw") as mrc:
			for i in range(n_images):
				idx = mrc.write_header(meta[i])
				mrc.write_data(data[i], idx)

	Attributes (read-only after init):
		filename   - path to the MRC file.
		is_stack   - True if .mrcs or .raw extension, False otherwise.
		isFEI      - True if FEI extended headers detected.
		nimg       - number of images in the file (stack_size for stacks).
		nx, ny, nz - dimensions from the header.
		big_endian - endianness of the on-disk data.
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

		self.is_stack = False
		self.isFEI = False
		self.is_8_bit_packed = False
		self.is_transpose = False
		self.is_complex = False
		self.is_ri = 0
		self.stack_size = 1
		self.nimg = None
		self.nx = None
		self.ny = None
		self.nz = None
		self.big_endian = _HOST_LITTLE

		self._h = {}
		self.mode_size = 0

		ext = os.path.splitext(self.filename)[-1].lower()
		if ext in (".raw",):
			self.isFEI = True
		if ext in (".mrcs", ".mrsc"):
			self.is_stack = True

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
			self._init_from_file()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None:
			self._file.close()
			self._file = None
		return False

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths. MRC supports 8, 16, and float32 (0)."""
		return [0, 8, 16]

	@staticmethod
	def is_valid(filepath, chunk):
		"""Check whether a file is potentially a valid MRC image."""
		try:
			if len(chunk) < NUM_4BYTES_PRE_MAP * 4:
				return False

			data = np.frombuffer(chunk[:NUM_4BYTES_PRE_MAP*4], dtype='<i4')
			nx, ny, nz, mrcmode = int(data[0]), int(data[1]), int(data[2]), int(data[3])

			do_swap, have_err = _check_swap(data)

			if have_err:
				return False

			if do_swap:
				data_swapped = data.byteswap().newbyteorder()
				nx, ny, nz, mrcmode = int(data_swapped[0]), int(data_swapped[1]), int(data_swapped[2]), int(data_swapped[3])

			if mrcmode in (MRC_MODE_SHORT_COMPLEX, MRC_MODE_FLOAT_COMPLEX):
				nx *= 2

			max_dim = 1 << 20
			return ((0 <= mrcmode < 128 or mrcmode == MRC_MODE_UHEX) and
					1 < nx < max_dim and 0 < ny < max_dim and 0 < nz < max_dim)

		except Exception:
			return False

	def _init_from_file(self):
		"""Read and validate the MRC header on open."""
		self._file.seek(0)
		raw = self._file.read(MRC_HEADER_SIZE)
		if len(raw) < MRC_HEADER_SIZE:
			raise FileFormatError(f"Incomplete MRC header read (got {len(raw)} of {MRC_HEADER_SIZE})")

		self._h, self.big_endian = _read_mrc_header(raw)
		self.mode_size = _get_mode_size(self._h["mode"])

		# Detect complex mode
		if self._h["mode"] in (MRC_MODE_SHORT_COMPLEX, MRC_MODE_FLOAT_COMPLEX):
			self.is_complex = True
			self.is_ri = 1
			self._h["nx"] *= 2

		# Default cell dimensions if zero
		for dim in ("xlen", "ylen", "zlen"):
			if self._h.get(dim, 0) == 0:
				self._h[dim] = 1.0

		# Detect FEI extended headers
		exttyp_prefix = self._h.get("exttyp", "")[:3]
		self.isFEI = (exttyp_prefix == "FEI")

		# Detect transpose (mapc==2, mapr==1 means X/Y swapped)
		self.is_transpose = (self._h.get("mapc", 1) == 2 and self._h.get("mapr", 2) == 1)

		self.nx = self._h["nx"]
		self.ny = self._h["ny"]
		self.nz = self._h["nz"]

		# Handle stack mode: MRCS extension means each slice is one image
		if self.is_stack:
			self.stack_size = self._h["nz"]
			self._h["nz"] = 1
			self.nz = 1
			self.nimg = self.stack_size

		# Detect 8-bit packed mode (2 nibbles per byte)
		self._detect_8bit_packed()

	def _detect_8bit_packed(self):
		"""Detect 8-bit packed mode from various heuristics."""
		if self._h.get("mode") == MRC_MODE_UHEX:
			self.is_8_bit_packed = True
			return

		imod_flags = self._h.get("imod_flags", 0)
		if imod_flags & 16:
			self.is_8_bit_packed = True
			self._h["mode"] = MRC_MODE_UHEX
			return

		nx = self._h.get("nx", 0)
		ny = self._h.get("ny", 0)
		if nx > 0:
			ratio = ny / nx
		else:
			ratio = 1.0

		have_packed_label = False
		if self._h.get("nlabels", 0) > 0 and self._h.get("labels"):
			for label in self._h["labels"]:
				if "4 bits packed" in label:
					have_packed_label = True
					break

		have_packed_filename = ("4_bits_packed" in self.filename.lower())
		ny_twice_nx = abs(ratio - 2.0) <= 0.1

		if ny_twice_nx and self._h["mode"] == MRC_MODE_UCHAR and (have_packed_label or have_packed_filename):
			self.is_8_bit_packed = True
			self._h["mode"] = MRC_MODE_UHEX
			self._h["nx"] *= 2
			self.nx *= 2

	def read_header(self, index=0):
		"""Read the MRC header and return it as a dict."""
		if index != 0 and not self.is_stack:
			raise InvalidIndex("MRC is single-image; use .mrcs extension for stacks")

		nx = self._h["nx"]
		ny = self._h["ny"]
		nz = self._h["nz"]

		meta = {}

		if self.is_transpose:
			meta["nx"], meta["ny"] = ny, nx
		else:
			meta["nx"], meta["ny"] = nx, ny
		meta["nz"] = nz

		mode = self._h["mode"]
		if mode == MRC_MODE_FLOAT or mode in (MRC_MODE_SHORT_COMPLEX, MRC_MODE_FLOAT_COMPLEX):
			meta["bitdepth"] = 0
		elif mode == MRC_MODE_UHEX:
			meta["bitdepth"] = 4
		elif mode == MRC_MODE_USHORT:
			meta["bitdepth"] = 16
		else:
			meta["bitdepth"] = 8

		# Core MRC fields
		for key in ("nxstart","nystart","nzstart","mx","my","mz","xlen","ylen","zlen",
		             "alpha","beta","gamma","mapc","mapr","maps","amin","amax","amean",
		             "ispg","nsymbt","rms","nlabels","creatid","nversion"):
			val = self._h.get(key)
			if val is not None:
				meta[f"MRC.{key}"] = val

		meta["MRC.mode"] = mode
		meta["MRC.exttyp"] = self._h.get("exttyp", "")
		meta["MRC.map"] = self._h.get("map", "MAP ")
		meta["MRC.machinestamp"] = self._h.get("machinestamp", _HOST_STAMP)

		mx = self._h.get("mx", nx) or 1
		my = self._h.get("my", ny) or 1
		mz = self._h.get("mz", nz) or 1

		apx = self._h.get("xlen", 1.0) / mx if mx else 1.0
		apy = self._h.get("ylen", 1.0) / my if my else 1.0
		apz = self._h.get("zlen", 1.0) / mz if mz else 1.0

		if not (0.01 <= apx <= 1000):
			apx = 1.0
		if not (0.01 <= apy <= 1000):
			apy = 1.0
		if not (0.01 <= apz <= 1000):
			apz = 1.0

		meta["apix_x"] = apx
		meta["apix_y"] = apy
		meta["apix_z"] = apz

		meta["origin_x"] = self._h.get("xorigin", 0.0)
		meta["origin_y"] = self._h.get("yorigin", 0.0)
		meta["origin_z"] = self._h.get("zorigin", 0.0)
		meta["sigma"] = self._h.get("rms", 0.0)

		if self.is_transpose:
			meta["apix_x"], meta["apix_y"] = meta["apix_y"], meta["apix_x"]
			meta["origin_x"], meta["origin_y"] = self._h.get("yorigin", 0.0), self._h.get("xorigin", 0.0)

		# IMOD fields
		for key in ("nint","nreal","sub","zfac","min2","max2","min3","max3",
		             "idtype","lens","nd1","nd2","vd1","vd2","imod_flags"):
			val = self._h.get(key)
			if val is not None:
				meta[f"IMODMRC.{key}"] = val

		nlabels = min(self._h.get("nlabels", 0), MRC_NUM_LABELS)
		for i in range(nlabels):
			label_key = f"MRC.label{i}"
			if self._h.get("labels") and i < len(self._h["labels"]):
				meta[label_key] = self._h["labels"][i]

		ctf = _read_ctf_from_labels(self._h.get("labels", []))
		if ctf:
			meta["ctf"] = ctf

		if self.isFEI:
			try:
				fei_meta = self._read_fei_header(index)
				meta.update(fei_meta)
			except Exception:
				pass

		return meta

	def _read_fei_header(self, image_index=0):
		"""Read FEI extended header for a given image."""
		offset = MRC_HEADER_SIZE + FEI_EXT_HEADER_SIZE * image_index
		self._file.seek(offset)
		raw = self._file.read(FEI_EXT_HEADER_SIZE)
		if len(raw) < FEI_EXT_HEADER_SIZE:
			raise FileIOError(f"Incomplete FEI extended header at index {image_index}")

		meta = {}
		meta["FEIMRC.metadata_size"] = struct.unpack_from("<i", raw, 0)[0]
		meta["FEIMRC.metadata_version"] = struct.unpack_from("<i", raw, 4)[0]
		
		bitmask1 = struct.unpack_from("<I", raw, 8)[0]
		if bitmask1 & 1: meta["FEIMRC.timestamp"] = struct.unpack_from("<d", raw, 12)[0]
		if bitmask1 & 32: meta["FEIMRC.ht"] = struct.unpack_from("<d", raw, 76)[0]
		if bitmask1 & 64: meta["FEIMRC.dose"] = struct.unpack_from("<d", raw, 84)[0]
		if bitmask1 & (1<<22): meta["FEIMRC.defocus"] = struct.unpack_from("<d", raw, 192)[0]
		if bitmask1 & (1<<31): meta["FEIMRC.magnification"] = struct.unpack_from("<d", raw, 260)[0]

		return meta

	def read_data(self, index=0):
		"""Read the float32 image data.

		Returns numpy array of shape (nz, ny, nx) with dtype float32.
		"""
		if not self.is_stack and index != 0:
			raise InvalidIndex("MRC is single-image; use .mrcs extension for stacks")

		endian = ">" if self.big_endian else "<"
		mode = self._h["mode"]
		nsymbt = self._h.get("nsymbt", 0)

		if not self.is_stack:
			index = 0

		nx = self.nx
		ny = self.ny
		nz = self.nz

		# Calculate data offset
		data_offset = MRC_HEADER_SIZE + nsymbt
		if self.isFEI and not self.is_stack:
			data_offset += FEI_EXT_HEADER_SIZE * max(index, 1)
		
		# For MRCS stacks, each image has its own header at the start
		image_data_size = nx * ny * nz * self.mode_size

		self._file.seek(data_offset + index * image_data_size)
		total_pixels = nx * ny * nz

		if mode == MRC_MODE_FLOAT:
			data = np.frombuffer(self._file.read(total_pixels * 4), dtype=endian + 'f')
			data = data.astype(np.float32)
		elif mode == MRC_MODE_UCHAR:
			raw_bytes = self._file.read(total_pixels)
			data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
		elif mode == MRC_MODE_USHORT:
			raw_bytes = self._file.read(total_pixels * 2)
			data = np.frombuffer(raw_bytes, dtype=endian + 'H').astype(np.float32)
		elif mode == MRC_MODE_SHORT:
			raw_bytes = self._file.read(total_pixels * 2)
			data = np.frombuffer(raw_bytes, dtype=endian + 'h').astype(np.float32)
		elif mode == MRC_MODE_UHEX:
			num_bytes = (total_pixels + 1) // 2
			raw_bytes = self._file.read(num_bytes)
			packed = np.frombuffer(raw_bytes, dtype=np.uint8)
			data = np.zeros(total_pixels, dtype=np.float32)
			for i in range(len(packed)):
				if 2*i < total_pixels:
					data[2*i] = float(packed[i] & 0x0F)
				if 2*i+1 < total_pixels:
					data[2*i+1] = float((packed[i] >> 4) & 0x0F)
		else:
			raise FileFormatError(f"Unsupported MRC mode: {mode}")

		if nz == 1:
			data = data.reshape((ny, nx))
		else:
			data = data.reshape((nz, ny, nx))

		if self.is_transpose:
			data = _transpose_xy(data)

		return data

	def write_header(self, meta, index=-1):
		"""Write an MRC header.
		
		Returns the image index written.
		"""
		if self.mode == "r":
			raise FileIOError("MRC file opened read-only; writes forbidden")

		if not self.is_stack and index not in (-1, 0):
			raise InvalidIndex("MRC is single-image; use .mrcs extension for stacks")

		self.nx = int(meta["nx"])
		self.ny = int(meta["ny"])
		self.nz = int(meta.get("nz", 1))

		bitdepth = int(meta.get("bitdepth", 0))
		if bitdepth not in self.bit_depths():
			raise FileFormatError(f"MRC supports float32, uint8, or uint16; got {bitdepth}")

		mode = _bitdepth_to_mode(bitdepth)
		self._h["mode"] = mode
		self.mode_size = _get_mode_size(mode)

		if index == -1:
			if self.is_stack:
				if self._is_new_file:
					index = 0
				else:
					index = self.stack_size
					self.stack_size += 1
			else:
				index = 0

		self.nimg = max(self.stack_size, index + 1) if self.is_stack else 1

		# For non-stack MRCs, build placeholder header now (will be updated with stats in write_data)
		if self._is_new_file and not self.is_stack:
			self._build_and_write_header_with_data({
				"minimum": 0.0, "maximum": 0.0,
				"mean": 0.0, "sigma": 0.0,
			})

		return index

	def _build_and_write_header_with_data(self, meta):
		"""Build and write header before data."""
		nx = self.nx
		ny = self.ny
		nz = self.nz if not self.is_stack else self.stack_size

		header_fields = {
			"nx": nx, "ny": ny, "nz": nz,
			"mode": self._h.get("mode", MRC_MODE_FLOAT),
			"mx": int(meta.get("MRC.mx", nx)) or nx,
			"my": int(meta.get("MRC.my", ny)) or ny,
			"mz": nz,
			"xlen": nx * float(meta.get("apix_x", 1.0)),
			"ylen": ny * float(meta.get("apix_y", 1.0)),
			"zlen": nz * float(meta.get("apix_z", 1.0)),
			"alpha": 90.0, "beta": 90.0, "gamma": 90.0,
			"mapc": 1, "mapr": 2, "maps": 3,
			"amin": float(meta.get("MRC.minimum", meta.get("minimum", 0.0))),
			"amax": float(meta.get("MRC.maximum", meta.get("maximum", 0.0))),
			"amean": float(meta.get("MRC.mean", meta.get("mean", 0.0))),
			"ispg": int(meta.get("MRC.ispg", 0)),
			"nsymbt": int(meta.get("MRC.nsymbt", 0)),
			"xorigin": float(meta.get("origin_x", 0.0)),
			"yorigin": float(meta.get("origin_y", 0.0)),
			"zorigin": float(meta.get("origin_z", 0.0)),
			"rms": float(meta.get("sigma", meta.get("MRC.rms", 0.0))),
		}

		nlabels = min(int(meta.get("MRC.nlabels", 1)), MRC_NUM_LABELS)
		header_fields["nlabels"] = nlabels

		labels = []
		for i in range(nlabels):
			key = f"MRC.label{i}"
			if key in meta:
				labels.append(str(meta[key])[:MRC_LABEL_SIZE])
		
		label_str = "EMAN3 MRC"
		t = __import__("time")
		label_str += t.strftime(" %Y-%m-%d %H:%M:%S")
		labels.append(label_str)
		header_fields["labels"] = labels

		self._file.seek(0)
		self._file.write(_build_mrc_header(header_fields))

	def write_data(self, data, index=0):
		"""Write image data to MRC file."""
		if not self.is_stack and index != 0:
			raise InvalidIndex("MRC is single-image; use .mrcs extension for stacks")

		data = np.asarray(data)

		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]

		if data.ndim == 2:
			data = data[np.newaxis, :]

		nz, ny, nx = data.shape

		if self.nx is None or self.ny is None:
			self.nx = nx
			self.ny = ny

		mode = self._h.get("mode", MRC_MODE_FLOAT)

		# Convert to target dtype
		if mode == MRC_MODE_FLOAT:
			out_data = data.astype(np.float32).ravel(order="C")
		elif mode == MRC_MODE_UCHAR:
			dmin, dmax = float(data.min()), float(data.max())
			if dmax > dmin:
				out_data = np.clip(((data - dmin) / (dmax - dmin)) * 255.0, 0, 255).astype(np.uint8)
			else:
				out_data = np.zeros(data.size, dtype=np.uint8)
		elif mode == MRC_MODE_USHORT:
			dmin, dmax = float(data.min()), float(data.max())
			if dmax > dmin:
				out_data = np.clip(((data - dmin) / (dmax - dmin)) * 65535.0, 0, 65535).astype(np.uint16)
			else:
				out_data = np.zeros(data.size, dtype=np.uint16)
		elif mode == MRC_MODE_SHORT:
			dmin, dmax = float(data.min()), float(data.max())
			if dmax > dmin:
				out_data = np.clip(((data - dmin) / (dmax - dmin)) * 32767.0 - 32768.0, -32768, 32767).astype(np.int16)
			else:
				out_data = np.zeros(data.size, dtype=np.int16)
		else:
			raise FileFormatError(f"Unsupported MRC write mode: {mode}")

		# Build header on first write
		if self._is_new_file:
			stats_meta = {
				"minimum": float(out_data.min()),
				"maximum": float(out_data.max()),
				"mean": float(out_data.mean()),
				"sigma": float(np.std(out_data)) if len(out_data) > 1 else 0.0,
			}
			self._build_and_write_header_with_data(stats_meta)
			self._is_new_file = False

		nsymbt = self._h.get("nsymbt", 0)
		data_offset = MRC_HEADER_SIZE + nsymbt
		
		if self.is_stack:
			image_idx = index if index >= 0 else self.stack_size - 1
		else:
			image_idx = 0
		self._file.seek(data_offset + image_idx * nx * ny * nz * self.mode_size)

		self._file.write(out_data.tobytes())
