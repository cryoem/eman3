#!/usr/bin/env python
#
# Author: Steven Ludtke,  (sludtke@bcm.edu)
# Copyright (c) 2000-  Baylor College of Medicine
#
# This software is issued under a joint BSD/GNU license. You may use the
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
#

#standard_library.install_aliases()  #not sure where this came from?
import sys
from math import *
from sys import exit
import os
import time
import shelve
import re
import pickle
import zlib
import socket
import subprocess
import EMAN3.EMAN3jsondb
from EMAN3.EMAN3jsondb import JSDict,js_open_dict,js_close_dict,js_remove_dict,js_list_dicts,js_check_dict,js_one_key
import argparse, copy
import glob
import random
from struct import pack,unpack
import json
from collections import OrderedDict
import traceback
from pathlib import Path
import numpy as np
import threading

__EMAN3=True
### If we ever need to add 'cleanup' exit code, this is how to do it. Drawn from the old BDB code.
## if the program exits nicely, close all of the databases
#atexit.register(DB_cleanup)

## if we are killed 'nicely', also clean up (assuming someone else doesn't grab this signal)
#signal.signal(2, DB_cleanup)
#signal.signal(15, DB_cleanup)

os.environ['QT_MAC_WANTS_LAYER'] = '1'

# This is a floating point number-finding regular expression
# Documentation: https://regex101.com/r/68zUsE/4/
#renumfind = re.compile(r"(?:^|(?<=[^\d\w:.-]))[+-]?\d+\.*\d*(?:[eE][-+]?\d+|)(?=[^\d\w:.-]|$)")
renumfind = re.compile(r"-?\d*\.?\d+[eE]?[+-]?\d*")


def e2gethome():
	"""platform independent path with '/'"""
	if (sys.platform != 'win32'):
		url = os.getenv("HOME")
	else:
		if (os.getenv("HOMEPATH") == '\\'):
			# Contributed by Alexander Heyne <AHEYNE@fmp-berlin.de>
			url = os.getenv("USERPROFILE")
			url = url.lstrip('CDEFG:')  # could also use substr
			url = url.replace("\\", "/")
		else:
			url = os.getenv("HOMEPATH")
			url = url.replace("\\", "/")
	return url

# This next line is to initialize the Transform object for threadsafety. Utterly stupid approach, but a functional hack
T=Transform({"type":"2d","alpha":0})

# When generating bispectral invariants, we need 2 parameters, which must be used consistently throughout the system
bispec_invar_parm=(32,10)

# These are processors which don't support in-place operation
outplaceprocs=["math.bispectrum.slice","math.harmonic","misc.directional_sum","morph.blackhat.binary","morph.close.binary","morph.dilate.binary","morph.erode.binary","morph.ext_grad.binary","morph.gradient.binary","morph.grow","morph.int_grad.binary","morph.majority","morph.object.density","morph.object.label","morph.open.binary","morph.prune","morph.thin","morph.tophat.binary"]

def cache_path():
	"""Returns the path to where .lsx file caches should be placed. Uses, in order:
	- EMAN3_CACHE_PATH env variable
	- project .json file
	- current folder
"""
	cp=os.getenv("EMAN3_CACHE_PATH")
	if cp is None:
		try:
			js=js_open_dict("info/project.json",false)
			cp=js["global.cache_path"]
		except: cp="cache/"

	if not os.path.exists(cp): os.makedirs(cp)

	return cp

# Without this, in many countries Qt will set things so "," is used as a decimal
# separator by sscanf and other functions, which breaks CTF reading and some other things
try:
	os.putenv("LC_CTYPE","en_US.UTF-8")
	os.putenv("LC_ALL","en_US.UTF-8")
except: pass

XYData.__len__=XYData.get_size

try:
	if __session__ is not None :
		GUIMode="jupyter"
		import ipykernel
except:
	try:
		if __IPYTHON__ : GUIMode="ipython"
		from PySide6 import QtGui, QtWidgets
		app=QtWidgets.QApplication.instance()
	except:
		GUIMode=None
		app = 0

GUIbeingdragged=None
originalstdout = sys.stdout

# This is to remove stdio buffering, only line buffering is done. This is what is done for the terminal, but this extends terminal behaviour to redirected stdio
# try/except is to prevent errors with systems that already redirect stdio
#try: sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
#except: pass

def stopautoflush():
	""" Return to buffered stdout """
	sys.stdout = originalstdout

# These are very widely used and hard to find, so some shortcuts
# Image file types
# IMAGE_MRC = EMUtil.ImageType.IMAGE_MRC
# IMAGE_EER = EMUtil.ImageType.IMAGE_EER
# IMAGE_EER2X = EMUtil.ImageType.IMAGE_EER2X
# IMAGE_EER4X = EMUtil.ImageType.IMAGE_EER4X
# IMAGE_SPIDER = EMUtil.ImageType.IMAGE_SPIDER
# IMAGE_SINGLE_SPIDER = EMUtil.ImageType.IMAGE_SINGLE_SPIDER
# IMAGE_IMAGIC = EMUtil.ImageType.IMAGE_IMAGIC
# IMAGE_HDF = EMUtil.ImageType.IMAGE_HDF
# IMAGE_DM3 = EMUtil.ImageType.IMAGE_DM3
# IMAGE_DM4 = EMUtil.ImageType.IMAGE_DM4
# IMAGE_TIFF = EMUtil.ImageType.IMAGE_TIFF
# IMAGE_PGM = EMUtil.ImageType.IMAGE_PGM
# IMAGE_LST = EMUtil.ImageType.IMAGE_LST
# IMAGE_PIF = EMUtil.ImageType.IMAGE_PIF
# IMAGE_VTK = EMUtil.ImageType.IMAGE_VTK
# IMAGE_PNG = EMUtil.ImageType.IMAGE_PNG
# IMAGE_SAL = EMUtil.ImageType.IMAGE_SAL
# IMAGE_ICOS = EMUtil.ImageType.IMAGE_ICOS
# IMAGE_EMIM = EMUtil.ImageType.IMAGE_EMIM
# IMAGE_GATAN2 = EMUtil.ImageType.IMAGE_GATAN2
# IMAGE_AMIRA = EMUtil.ImageType.IMAGE_AMIRA
# IMAGE_XPLOR = EMUtil.ImageType.IMAGE_XPLOR
# IMAGE_EM = EMUtil.ImageType.IMAGE_EM
# IMAGE_V4L = EMUtil.ImageType.IMAGE_V4L
# IMAGE_UNKNOWN = EMUtil.ImageType.IMAGE_UNKNOWN

# # image data storage modes
# EM_UNKNOWN = EMUtil.EMDataType.EM_UNKNOWN
# EM_CHAR = EMUtil.EMDataType.EM_CHAR
# EM_UCHAR = EMUtil.EMDataType.EM_UCHAR
# EM_SHORT = EMUtil.EMDataType.EM_SHORT
# EM_USHORT = EMUtil.EMDataType.EM_USHORT
# EM_INT = EMUtil.EMDataType.EM_INT
# EM_UINT = EMUtil.EMDataType.EM_UINT
# EM_FLOAT = EMUtil.EMDataType.EM_FLOAT
# EM_DOUBLE = EMUtil.EMDataType.EM_DOUBLE
# EM_SHORT_COMPLEX = EMUtil.EMDataType.EM_SHORT_COMPLEX
# EM_USHORT_COMPLEX = EMUtil.EMDataType.EM_USHORT_COMPLEX
# EM_FLOAT_COMPLEX = EMUtil.EMDataType.EM_FLOAT_COMPLEX
# EM_COMPRESSED = EMUtil.EMDataType.EM_COMPRESSED


# These map standard names for data types to internal representation, and provide a minimum and maximum value for each type
# file_mode_map={
# 	"int8"  :EMUtil.EMDataType.EM_CHAR,
# 	"uint8" :EMUtil.EMDataType.EM_UCHAR,
# 	"int16" :EMUtil.EMDataType.EM_SHORT,
# 	"uint16":EMUtil.EMDataType.EM_USHORT,
# 	"int32" :EMUtil.EMDataType.EM_INT,
# 	"uint32":EMUtil.EMDataType.EM_UINT,
# 	"float" :EMUtil.EMDataType.EM_FLOAT,
# 	"compressed": EMUtil.EMDataType.EM_COMPRESSED }

# inverse dictionary for getting printable names
# file_mode_imap=dict([[int(v),k] for k,v in list(file_mode_map.items())])

# file_mode_intmap={
# 	1 :EMUtil.EMDataType.EM_CHAR,
# 	2 :EMUtil.EMDataType.EM_UCHAR,
# 	3 :EMUtil.EMDataType.EM_SHORT,
# 	4 :EMUtil.EMDataType.EM_USHORT,
# 	5 :EMUtil.EMDataType.EM_INT,
# 	6 :EMUtil.EMDataType.EM_UINT,
# 	7 :EMUtil.EMDataType.EM_FLOAT,
# 	8 :EMUtil.EMDataType.EM_COMPRESSED }


#keyed both by type and by the integer version for flexibility
# file_mode_range={
# 	EMUtil.EMDataType.EM_CHAR:(-128,127),
# 	EMUtil.EMDataType.EM_UCHAR:(0,255),
# 	EMUtil.EMDataType.EM_SHORT:(-32768,32767 ),
# 	EMUtil.EMDataType.EM_USHORT:(0,65535 ),
# 	EMUtil.EMDataType.EM_INT:(-2147483648,2147483647 ),
# 	EMUtil.EMDataType.EM_UINT:(0,4294967295),
# 	EMUtil.EMDataType.EM_FLOAT:(-3.40282347e+38,3.40282347e+38 ),
# 	int(EMUtil.EMDataType.EM_CHAR):(-128,127),
# 	int(EMUtil.EMDataType.EM_UCHAR):(0,255),
# 	int(EMUtil.EMDataType.EM_SHORT):(-32768,32767 ),
# 	int(EMUtil.EMDataType.EM_USHORT):(0,65535 ),
# 	int(EMUtil.EMDataType.EM_INT):(-2147483648,2147483647 ),
# 	int(EMUtil.EMDataType.EM_UINT):(0,4294967295),
# 	int(EMUtil.EMDataType.EM_FLOAT):(-3.40282347e+38,3.40282347e+38 ),
# 	int(EMUtil.EMDataType.EM_COMPRESSED):(-3.40282347e+38,3.40282347e+38 )
# 	}

class NotImplementedException(Exception):
	def __init__(self,val=None): pass

class FileFormatError(Exception):
	"""Exception for problems with File formats"""
	# pass

class FileIOError(Exception):
	"""Exception for problems with File IO"""
	pass

prog_t0=0
def print_progress(tlast,prefix,cur,n):
	"""Prints/updates a progress message on the console based on elapsed time since tlast. Returns the appropriate tlast for the next
	call. This can be used for overall progress of the program or for progress in individual stages. This is a stateful function, not
	appropriate for threading, and MUST be called with cur==n at the end to reset for the next use in the current kernel."""
	global prog_t0

#	print(prog_t0,tlast,prefix,cur,n)
	if cur>=n:
		prog_t0=0
		print(f"\n{prefix} complete")
		return 0

	if prog_t0==0:
		prog_t0=time.time()
		return tlast

	ct=time.time()
	if ct-tlast<2 : return tlast
	if cur>0 : eta=int((ct-prog_t0)/(cur/n)-(ct-prog_t0))  # estimated time remaining
	else : eta=9999*60
	print(f"  {prefix}: {cur/n*100:3.1f}% {eta//60:02d}:{eta%60:02d} ETA   ",end="\r")
	sys.stdout.flush()
	return ct

def E3init(argv, ppid=-1) :
	"""E3init(argv)
This function is called to log information about the current job to the local logfile. The flags stored for each process
are pid, start, args, progress and end. progress is from 0.0-1.0 and may or may not be updated. end is not set until the process
is complete. If the process is killed, 'end' may never be set."""

	# We go to the end of the file. Record the location, then write a fixed length string
	try:
		hist=open(".eman3log.txt","r+")
		hist.seek(0,os.SEEK_END)
	except:
		try: hist=open(".eman3log.txt","w")
		except: return -1
	n=hist.tell()
	hist.write(f"{local_datetime()}\t{'0':19s}\t{'-':11s}\t{os.getpid()}/{ppid}\t{socket.gethostname()}\t{' '.join(argv)}\n")

	hist.close()

	return n

def E3progress(n,progress):
	"""Updates the progress fraction (0.0-1.0) for a running job. Negative values may optionally be
set to indicate an error exit."""
#	if EMAN2db.BDB_CACHE_DISABLE : return		# THIS MUST REMAIN DISABLED NOW THAT THE CACHE IS DISABLED PERMANENTLY !!!

	try:
		hist=open(".eman3log.txt","r+")
		hist.seek(n+20)
	except:
		return -1
	hist.write(f"{int(progress*100.0):19.0f}")
#	hist.flush()
	hist.close()

	return n

def E3end(n):
	"""E3end(n)
This function is called to log the end of the current job. n is returned by E3init"""
	try:
		hist=open(".eman3log.txt","r+")
		hist.seek(n)
		start=hist.read(19)
		end=local_datetime()
	except:
		return -1
	hist.seek(n+20)
	try: hist.write(f"{end:19s}\t{timestamp_diff(start,end):11d}")
	except: print(f"E3end failed: {start} {end} {n}")
	hist.close()

	return n

def E3saveappwin(app,key,win):
	"""stores the window geometry using the application default mechanism for later restoration. Note that
	this will only work with Qt windows"""
	try:
		pos=win.pos()
		sz=win.size()
		geom=(pos.x(),pos.y(),win.width(),win.height())

		E2setappval(app,key,geom)
#		print(app,key,geom)
	except:
		print("Error saving window location ",key)

def E3loadappwin(app,key,win):
	"""restores a geometry saved with E2saveappwin"""
	try:
		geom=list(E2getappval(app,key))
		if geom==None : raise Exception
		win.resize(geom[2],geom[3])
		geom[0]=max(32,geom[0])
		geom[1]=max(60,geom[1])
		win.move(geom[0],geom[1])
#		print(app,key,geom)
	except: return

def E3setappval(app,key,value):
	"""E2setappval
This function will set an application default value both in the local directory and ~/.eman2
When settings are read, the local value is checked first, then if necessary, the global value."""
	try:
		app.replace(".","_")
		key.replace(".","_")
	except:
		print("Error with E2setappval, app and key must be strings")
		return

	try:
		db=js_open_dict(".eman2settings.json")
		db[app+"."+key]=value
#		db.close()
	except:
		pass

	try:
		dir=e2gethome()
		dir+="/.eman2"
	except:
		return
	
	try: os.mkdir(dir)
	except: pass

	try:
		db=js_open_dict(dir+"/eman2settings.json")
		db[app+"."+key]=value
#		db.close()
	except:
		return


def E3getappval(app,key,dfl=None):
	"""E2getappval
This function will get an application default by first checking the local directory, followed by
~/.eman2"""
	try:
		app.replace(".","_")
		key.replace(".","_")
	except:
		print("Error with E2getappval, app and key must be strings")
		return None

	try:
		db=js_open_dict(".eman2settings.json")
		ret=db[app+"."+key]
		return ret
	except:
		pass

	try:
		dir=e2gethome()
		dir+="/.eman2"
		db=js_open_dict(dir+"/eman2settings.json")
		ret=db[app+"."+key]
		db.close()

		return ret
	except: pass

	if dfl!=None: E2setappval(app,key,dfl)		# so the user will know what may be available

	return dfl

def E3getappvals():
	"""E2getappvals
This function will return a list of lists containing all currently set application defaults as [program,option,value,global|local]"""

	ret=[]
	ret2=[]

	try:
		db=js_open_dict(".eman2settings.json")
		keys=db.keys()
		ret=[(k.split(".")[0],k.split(".")[1],db[k],"local") for k in keys]
	except:
		pass
	
	try:
		dir=e2gethome()
		dir+="/.eman2"
		dbu=js_open_dict(dir+"/eman2settings.json")
		ret2=[(k.split(".")[0],k.split(".")[1],dbu[k],"user") for k in dbu.keys() if k not in keys]  # only show global when local doesn't exist
		
	except: pass

	return ret2+ret

def get_temp_name():
	"""Returns a suitable name for a temporary HDF file in the current directory. Does not create or delete the file."""
	fsp=f"tmp_{random.randint(0,9999999):07d}.hdf"
	if os.path.exists(fsp) : return get_temp_name()		# risky? shouldn't really ever recurse forever...
	return fsp

def num_path_new(prefix):
	"""make a new prefix_xx (or prefix_xxx) folder and return the folder name, underscore added to prefix if not present"""
	if prefix[-1]!="_" : prefix+="_"
	
	pthns=[int(i.rsplit("_",1)[-1]) for i in os.listdir(".") if i[:len(prefix)]==prefix and i.rsplit("_",1)[-1].isdigit()]
	try: newn=max(pthns)+1
	except: newn=0
	path=f"{prefix}{newn:02d}"
	try: os.mkdir(path)
	except:
		raise Exception("Error: could not create "+path)
	
	return path

def num_path_last(prefix,create=False):
	"""find the highest numbered path starting with prefix and return it. If create is set, will create a new one if none exists. underscore added to prefix if not present"""
	if prefix[-1]!="_" : prefix+="_"
	
	pthns=[int(i.rsplit("_",1)[-1]) for i in os.listdir(".") if i[:len(prefix)]==prefix and i.rsplit("_",1)[-1].isdigit()]
	try: n=max(pthns)
	except:
		if create: 
			path=f"{prefix}00"
			try: os.mkdir(path)
			except: raise Exception("Error: could not create "+path)
			return path
		else: raise Exeception("Error: could not find paths beginning with "+prefix)
	
	path=f"{prefix}{n:02d}"
	
	return path

# since there are only 3 odd numbers in the entire list, we remove them, and just limit ourselves to even numbers
# Values thru 512 are carefully calculated as shown on the wiki. Larger values have prime factors 7 or lower.
# 9/2/17 removing low numbers not divisible by 4 for convenience
good_box_sizes=[16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 84, 96, 100, 104, 112, 120, 128, 132, 140, 168, 180, 192, 196, 208, 216, 220, 224, 240, 256, 260, 288, 300, 320, 352, 360, 384, 416, 440, 448, 480, 512, 540, 560, 576, 588, 600, 630, 640, 648, 672, 686, 700, 720, 750, 756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1050, 1080, 1120, 1134, 1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512, 1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800, 1890, 1920, 1944, 1960, 2000, 2016, 2048, 2058, 2100, 2160, 2240, 2250, 2268, 2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688, 2700, 2744, 2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402, 3430, 3456, 3500, 3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032, 4050, 4096, 4116, 4200, 4320, 4374, 4410, 4480, 4500, 4536, 4608, 4704, 4800, 4802, 4860, 4900, 5000, 5040, 5120, 5184, 5250, 5292, 5376, 5400, 5488, 5600, 5670, 5760, 5832, 5880, 6000, 6048, 6144, 6174, 6250, 6272, 6300, 6400, 6480, 6720, 6750, 6804, 6860, 6912, 7000, 7056, 7168, 7200, 7290, 7350, 7500, 7560, 7680, 7776, 7840, 7938, 8000, 8064, 8100, 8192, 8232, 8400, 8640, 8748, 8750, 8820, 8960, 9000, 9072, 9216, 9408, 9450, 9600, 9604, 9720, 9800, 10000, 10080, 10206, 10240, 10290, 10368, 10500, 10584, 10752, 10800, 10976, 11200, 11250, 11340, 11520, 11664, 11760, 12000, 12096, 12150, 12250, 12288, 12348, 12500, 12544, 12600, 12800, 12960, 13122, 13230, 13440, 13500, 13608, 13720, 13824, 14000, 14112, 14336, 14400, 14406, 14580, 14700, 15000, 15120, 15360, 15552, 15680, 15750, 15876, 16000, 16128, 16200, 16384]

def good_size(size):
	"""Will return the next larger 'good' box size (with good refinement performance)"""
#	sizes=[32, 33, 36, 40, 42, 44, 48, 50, 52, 54, 56, 60, 64, 66, 70, 72, 81, 84, 96, 98, 100, 104, 105, 112, 120, 128,130, 132, 140, 150, 154, 168, 180, 182, 192, 196, 208, 210, 220, 224, 240, 250, 256,260, 288, 300, 320, 330, 352, 360, 384, 416, 440, 448, 450, 480, 512]
	# since there are only 3 odd numbers in the entire list, we remove them, and just limit ourselves to even numbers
	for i in good_box_sizes :
		if i>=size: return i

	return Util.calc_best_fft_size(int(size))

def good_size_small(size):
	"""Will return the next larger 'good' box size (with good refinement performance)"""
#	sizes=[32, 33, 36, 40, 42, 44, 48, 50, 52, 54, 56, 60, 64, 66, 70, 72, 81, 84, 96, 98, 100, 104, 105, 112, 120, 128,130, 132, 140, 150, 154, 168, 180, 182, 192, 196, 208, 210, 220, 224, 240, 250, 256,260, 288, 300, 320, 330, 352, 360, 384, 416, 440, 448, 450, 480, 512]

	for i in reversed(good_box_sizes) :
		if i<=size: return i

def commandoptions(options,exclude=[]):
	"""This will reconstruct command-line options, excluding any options in exclude"""
	opts=[]
	for opt,val in vars(options).items():
		if opt in exclude or opt=="positionalargs" or val is False or val==None: continue
		if val==True and isinstance(val,bool) : opts.append("--"+opt)
		else: opts.append("--{}={}".format(opt,val))

	return " ".join(opts)


class EMArgumentParser(argparse.ArgumentParser):
	""" subclass of argparser to masquerade as optparser and run the GUI """
	def __init__(self, prog=None,usage=None,description=None,epilog=None,version=None,parents=[],formatter_class=argparse.HelpFormatter,prefix_chars='-',fromfile_prefix_chars=None,argument_default=None,conflict_handler='error',add_help=True,allow_abbrev=True):
		argparse.ArgumentParser.__init__(self,prog=prog,usage=usage,description=description,epilog=epilog,parents=parents,formatter_class=formatter_class,prefix_chars=prefix_chars,fromfile_prefix_chars=fromfile_prefix_chars,argument_default=argument_default,conflict_handler=conflict_handler,add_help=add_help,allow_abbrev=allow_abbrev)

		# A list of options to add to the GUI
		self.optionslist = []
		self.tablist = []

		if version:
			self.add_argument('--version', action='version', version=version)
		self.add_argument("positionalargs", nargs="*")
		self.add_argument('--help-to-html', action='store_true',
		                  help='print this help message in html format')

	def parse_args(self):
		""" Masquerade as optparser parse options """
		if "--help-to-html" in sys.argv[1:]:
			import pandas as pd
			from contextlib import redirect_stdout
			from io import StringIO

			actions = self._get_optional_actions()

			df = pd.DataFrame(columns=['Option', 'Type', 'Description'])

			for i in actions:
				i.type = "None" if not i.type else str(i.type).split("'")[1]
				i.option_strings = ', '.join(i.option_strings)

				if "--help-to-html" in i.option_strings or "--help" in i.option_strings:
					continue

				df = pd.concat([df, pd.DataFrame({'Option': [i.option_strings],
				                                  'Type': [i.type],
				                                  'Description': [i.help]})],
				               ignore_index=True)

			print('<pre>')

			stdout = StringIO()
			with redirect_stdout(stdout):
				self.print_usage()
			print(stdout.getvalue().replace('<', '&lt').replace('>', '&gt'))

			print('</pre>\n')

			print(df.to_html(index=False, justify="center"))

			self.exit()

		parsedargs = argparse.ArgumentParser.parse_args(self)

		return (parsedargs, parsedargs.positionalargs)

	def add_pos_argument(self, **kwargs):
		""" Add a position argument, needed only for the GUI """
		kwargs["positional"]=True
		self.optionslist.append(copy.deepcopy(kwargs))

	def add_header(self, **kwargs):
		""" for the header, you need, title, row, col"""
		kwargs["guitype"]="header"
		self.optionslist.append(copy.deepcopy(kwargs))

	def add_argument(self, *args, **kwargs):

		if "guitype" in kwargs:
			if args[0][:2] == "--":
				kwargs["name"] = args[0][2:]
			else:
				kwargs["name"] = args[1][2:]
			self.optionslist.append(copy.deepcopy(kwargs))
			del kwargs["guitype"]
			del kwargs["row"]
			del kwargs["col"]
			del kwargs["name"]
			if "rowspan" in kwargs: del kwargs["rowspan"]
			if "colspan" in kwargs: del kwargs["colspan"]
			if "expert" in kwargs: del kwargs["expert"]
			if "lrange" in kwargs: del kwargs["lrange"]
			if "urange" in kwargs: del kwargs["urange"]
			if "choicelist" in kwargs: del kwargs["choicelist"]
			if "filecheck" in kwargs: del kwargs["filecheck"]
			if "infolabel" in kwargs: del kwargs["infolabel"]
			if "mode" in kwargs: del kwargs["mode"]
			if "browser" in kwargs: del kwargs["browser"]
			if "dirbasename" in kwargs: del kwargs["dirbasename"]
			if "nosharedb" in kwargs: del kwargs["nosharedb"]
			if "returnNone" in kwargs: del kwargs["returnNone"]

		argparse.ArgumentParser.add_argument(self, *args, **kwargs)

	def getGUIOptions(self):
		return self.optionslist

#	# FIXME - this function is no longer necessary since I overwrite the Symmetry3D::get function (on the c side). d.woolford
def parsesym(optstr):
	return Symmetries.get(optstr)

parseparmobj1=re.compile(r"([^\(]*)\(([^\)]*)\)")	# This parses test(n=v,n2=v2) into ("test","n=v,n2=v2")
parseparmobj2=re.compile(r"([^=,]*)=([^,]*)")		# This parses "n=v,n2=v2" into [("n","v"),("n2","v2")]
parseparmobj3=re.compile(r"[^:]\w*=*[-\w.]*") # This parses ("a:v1=2:v2=3") into ("a", "v1=2", "v2=3")
parseparmobj4=re.compile(r"\w*[^=][\w.]*") # This parses ("v1=2") into ("v1", "2")

def parse_transform(optstr):
	"""This is used so the user can provide the rotation information and get the transform matrix in a convenient form.
	It will parse "dot0,dot1,dot2" and type:name=val:name=val:... then return the tranform matrix t=transform(az,alt,phi)"""

	# We first check for the 'EMAN-style' special case
	tpl=optstr.split(",")
	if len(tpl)==3 :
		try: tpl=[float(i) for i in tpl]
		except:
			raise Exception("Invalid EMAN transform: %s"%optstr)
		return Transform({"type":"eman","az":tpl[0],"alt":tpl[1],"phi":tpl[2]})

	# Now we must assume that we have a type:name=val:... specification
	tpl=optstr.split(":")
	parms={"type":tpl[0]}

	# loop over parameters and add them to the dictionary
	for parm in tpl[1:]:
		s=parm.split("=")
		try : parms[s[0]]=float(s[1])
		except :
			raise Exception("Invalid transform parameter: %s"%parm)

	try: ret=Transform(parms)
	except:
		raise Exception("Invalid transform: %s"%optstr)

	return ret

def unparsemodopt(tupl):
	"""This takes a 2-tuple of the form returned by parsemodopt and returns a corresponding string representation"""

	try:
		if tupl[0]==None : return ""
		if tupl[1]==None or len(tupl[1])==0 : return str(tupl[0])
		parm=["{}={}".format(k,v) for k,v in list(tupl[1].items())]
		return str(tupl[0])+":"+":".join(parm)
	except:
		return ""

def parsemodopt(optstr=None):
	"""This is used so the user can provide the name of a comparator, processor, etc. with options
	in a convenient form. It will parse "dot:normalize=1:negative=0" and return
	("dot",{"normalize":1,"negative":0})"""

	if optstr is None or len(optstr)==0 : return (None,{})
	if optstr.lower()=="none" : return None					# special case doesn't return a tuple

	op2=optstr.split(":")
	if len(op2)==1 or op2[1]=="" : return (op2[0],{})		# name with no options

	r2={}
	for p in op2[1:]:
		try: k,v=p.split("=")
		except:
			print("ERROR: Command line parameter parsing failed on ",optstr)
			print("must have the form name:key=value:key=value")
			return(None,None)

#		v=v.replace("bdb%","bdb:")
		if v.lower()=="true" : v=1
		elif v.lower()=="false" : v=0
		else:
			try: v=int(v)
			except:
				try: v=float(v)
				except:
					if len(v)>2 and v[0]=='"' and v[-1]=='"' : v=v[1:-1]
		r2[k]=v

	return (op2[0],r2)

def parsedict(dictstr):
	cmpparms = {}
	for d in dictstr:
		keyval = d.split(":")
		try:
			cmpparms[keyval[0]] = float(keyval[1])
		except:
			cmpparms[keyval[0]] = keyval[1]
	return cmpparms

parseparmobj_op = re.compile(r"\+=|-=|\*=|\/=|%=")
parseparmobj_logical = re.compile(r">=|<=|==|~=|!=|<|>") 	# finds the logical operators <=, >=, ==, ~=, !=, <, >
parseparmobj_op_words = re.compile(r"\w*[^=+-\/\*%][\w.]*") # splits ("v1?2") into ("v1","2") where ? can be any combination of the characters "=!<>~"
parseparmobj_logical_words = re.compile(r"\w*[^=!<>~][\w.]*") # splits ("v1?2") into ("v1","2") where ? can be any combination of the characters "=!<>~"
def parsemodopt_logical(optstr):

	if not optstr or len(optstr)==0 : return (None)

	p_1 = re.findall( parseparmobj_logical_words, optstr )

	if len(p_1)==0: return (optstr,{})
	if len(p_1) != 2:
		print("ERROR: parsemodopt_logical currently only supports single logical expressions")
		print("Could not handle %s" %optstr)
		return (None,None,None)

	p_2 = re.findall( parseparmobj_logical, optstr )

	if len(p_2) != 1:
		print("ERROR: could not find logical expression in %s" %optstr)
		return (None,None,None)


	if ( p_2[0] not in ["==", "<=", ">=", "!=", "~=", "<", ">"] ):
		print("ERROR: parsemodopt_logical %s could not extract logical expression" %(p_2[0]))
		print('Must be one of "==", "<=", ">=", "<", ">" "!=" or "~=" ')
		return (None,None,None)

	return p_1[0], p_2[0], p_1[1]


def parsemodopt_operation(optstr):

	if not optstr or len(optstr)==0 : return (None)

	p_1 = re.findall( parseparmobj_op_words, optstr )
	if len(p_1)==0: return (optstr,{})

	if len(p_1) != 2:
		print("ERROR: parsemodopt_logical currently only supports single logical expressions")
		print("Could not handle %s" %optstr)
		return (None,None,None)

	p_2 = re.findall( parseparmobj_op, optstr )
	if len(p_2) != 1:
		print("ERROR: could not find logical expression in %s" %optstr)
		return (None,None,None)


	if ( p_2[0] not in ["+=", "-=", "*=", "/=", "%="]):
		print("ERROR: parsemodopt_logical %s could not extract logical expression" %(p_2[0]))
		print("Must be one of", "+=", "-=", "*=", "/=", "%=")
		return (None,None,None)

	return (p_1[0], p_2[0], p_1[1])

def read_number_file(path):
	"""This will read a text file containing a list of integers. The integers may be separated by any character(s). Any contiguous
	sequence of (0-9) in the file will be treated as a number. '#' is NOT respected as a comment character."""

	try:
		regex = re.compile(r"[0-9]+")
		return [int(i) for i in regex.findall(open(path,"r").read())]
	except:
		return []


def parse_list_arg(*possible_types):
	"""Return a function that can be passed to argparse.add_argument() as an argument to 'type'.
	The returned function's return type is a list of instances in possible_types, i.e., the parsed
	arguments are converted to the types given in possible_types.

	>>> parse_list_arg(int,float)("3, 4")
	[3, 4.0]
	>>> parse_list_arg(int,int)("3, 4")
	[3, 4]
	>>> parse_list_arg([int,int],[int,int,int])("3,4,5")
	[3, 4, 5]
	>>> parse_list_arg([int,int],[int,int,int])("3,4")
	[3, 4]
	>>> parse_list_arg([int,str],[int,int,int])("3,4")
	[3, '4']
	>>> parse_list_arg([int,str],[int,int,int])("3,4,5")
	[3, 4, 5]
	>>> parse_list_arg([int],[int,int],[int,int,int])("3,4")
	[3, 4]
	>>> parse_list_arg([int],[int,int],[int,int,int])("3")
	[3]
	"""

	types_dict = {}

	# If a single possible set of types is given, they don't have to be wrapped in a list/tuple
	# To determine that, the first item in possible_types is checked if it is a list/tuple instance
	if not isinstance(possible_types[0], (tuple, list)):
		types_dict[len(possible_types)] = possible_types
	else:
		for i in possible_types:
			types_dict[len(i)] = i

	# This is the function that will be passed to argparse.add_argument()'s 'type' argument
	def arg_to_list(s):
		user_input_str = s.split(',')

		if not any(len(user_input_str) == k for k in types_dict.keys()):
			raise argparse.ArgumentTypeError(f"provide {' or '.join(str(i) for i in types_dict.keys())} arguments! See --help for details.")
		else:
			types = types_dict[len(user_input_str)]

		return [type_to_convert(usr_input_val)
				for type_to_convert, usr_input_val
				in zip(types, user_input_str)]

	return arg_to_list


def parse_string_to_slices(seq):
	"""
	>>> parse_string_to_slices("")
	[slice(None, None, None)]
	>>> parse_string_to_slices("3,4")
	[3, 4]
	>>> parse_string_to_slices("3,4,2:5")
	[3, 4, slice(2, 5, None)]
	>>> parse_string_to_slices("3,4,2:5:")
	[3, 4, slice(2, 5, None)]
	>>> parse_string_to_slices("3,4,2:5:2")
	[3, 4, slice(2, 5, 2)]
	>>> parse_string_to_slices("3,4,2:5:2,8,14")
	[3, 4, slice(2, 5, 2), 8, 14]
	>>> parse_string_to_slices("3,4,2::2,8,14")
	[3, 4, slice(2, None, 2), 8, 14]
	>>> parse_string_to_slices("3,4,::2,8,14")
	[3, 4, slice(None, None, 2), 8, 14]
	>>> parse_string_to_slices("3,4,:11:2,8,14")
	[3, 4, slice(None, 11, 2), 8, 14]
	"""

	if not seq:
		return [slice(None, None, None)]

	seq = filter(None, seq.split(','))
	slices = []

	for s in seq:
		if s.isnumeric():
			slices.append(int(s))
		elif ":" in s:
			sl = [None] * 3
			for i, k in enumerate(s.split(":")):
				sl[i] = int(k) if len(k) > 0 else None
			slices.append(slice(*sl))
		else:
			with open(s) as fin:
				file_content = fin.read().replace('\n', ',').replace(' ', ',')

			slices.extend(parse_string_to_slices(file_content))

	return slices


def parse_infile_arg(arg):
	"""
	Parses a string of file name with inclusion and exclusion lists
	Returns the file name and a list of image indices

	Format of the input string:

	filename:inclusion_list^exclusion_list

	inclusion_list/exclusion_list may contain comma-separated
	1. integers
	2. slices (Python's built-in slice representation)
	3. file names

	No spaces are allowed in the passed string.
	Negative integers are allowed as in Python slices.
	However, listed files may contain multi-lines and may contain spaces.

	Slice examples for an image file with N images where N>20:
	2:    -> 2,3,...,N-1
	:3:   -> 0,1,2
	::5   -> 0,5,10,15,...,<N
	:10:2 -> 0,2,4,6,8
	::-1  -> N-1,N-2,N-3,...,2,1,0
	2:9   -> 2,3,4,5,6,7,8

	If a file named test_images.hdf contains 100 images,
	"test_images.hdf:2,4:11:,9,6^25:28,30" will be interpreted as follows.
	filename: test_images.hdf
	inclusion_list: 2,4:11:,19,26:32:2 -> 2,3,4,5,6,7,8,9,10,19,26,28,30
	exclusion_list: 25:28,30      -> 25,26,27,30

	Returned index list: [2,3,4,5,6,7,8,9,10,19,26,28,30] - [25,26,27,30] ->
					   : [2,3,4,5,6,7,8,9,10,19,28]
	"""

	fname, _, seq = arg.partition(':')

	if not (fname and os.path.isfile(fname)):
		raise Exception(f"'{fname}' is not an existing regular file!")

	seq_inc, _, seq_exc = seq.partition('^')

	slices_inc = parse_string_to_slices(seq_inc)
	slices_exc = parse_string_to_slices(seq_exc) if seq_exc else []

	nimg = file_image_count(fname)
	if ":" not in arg: return arg,range(nimg)		#quick stopgap for performance problem

	idxs = OrderedDict()
	for i in slices_inc:
		if isinstance(i, int):
			idxs.update({i: None})
		else:
			for k in range(*(i.indices(nimg))):
				idxs.update({k: None})

	ids_exc = set()
	for i in slices_exc:
		if isinstance(i, int):
			ids_exc.add(i)
		else:
			ids_exc.update(range(*(i.indices(nimg))))

	for i in ids_exc:
		if i in idxs:
			idxs.pop(i)

	return fname, tuple(idxs.keys())

def parse_range(rangestr,maxval=None):
	"""parses strings like "1,3,4,6-9,11" and returns (1,3,4,6,7,8,9,11), maxval will support n- without upper values
	if not provided will still work as long as n- isn't used"""

	ret=[]
	for s in rangestr.split(","):
		try: ret.append(int(s))
		except:
			try:
				v1,v2=s.split("-")
				ret.extend(range(int(v1),int(v2)+1))
			except:
				v1=int(s.split("-")[0])
				ret.extend(range(v1,maxval+1))

	return ret


def parse_outfile_arg(arg):
	"""
	Input:
	<filename>:<outbits>[:f|o|<min>|<min>s:<max>|<max>s]
	f - full range
	o - full range with extreme outlier removal
	<min> is absolute min value
	<min>s mean-min*sigma
	if only outbits specified, "o" is implied
	if bits==0 -> floating point with lossless compression
	if bits<0 -> floating point with no compression
	returns:
	(filename,outbits,f|o|None,<min>,<max>,<mins>,<maxs>)
	filename:outbits:rendermin:rendermax

	out.hdf:6:20:1000
	out.hdf:6:-3s:5s
	out.hdf:6:f
	out.hdf:6

	"""

	parm=arg.split(":")
	if len(parm)==0 or len(parm[0])==0:
		raise Exception("Output filename required")

	try: parm[1]=int(parm[1])
	except:	raise Exception("Output filename <filename>:<bits>")

	# only bits specified, default behavior is "o"
	if len(parm)==2: return(parm[0],parm[1],"o",None,None,None,None)

	# special modes
	if   parm[2].lower()=="f" : return(parm[0],parm[1],"f",0,0,0,0)
	elif parm[2].lower()=="o" : return(parm[0],parm[1],"o",0,0,0,0)
	
	# no special mode, either values or sigma coefficients
	vals=[None,None,None,None]
	if "s" in parm[2]:
		try: vals[2]=float(parm[2][:-1])
		except: raise Exception("Output filename: <filename>:<outbits>[:f|o|<min>|<min>s[:<max>|<max>s]]")
	else:
		try: vals[0]=float(parm[2])
		except: raise Exception("Output filename: <filename>:<outbits>[:f|o|<min>|<min>s[:<max>|<max>s]]")

	if "s" in parm[3]:
		try: vals[3]=int(parm[3][:-1])
		except: raise Exception("Output filename: <filename>:<outbits>[:f|o|<min>|<min>s[:<max>|<max>s]]")
	else:
		try: vals[0]=int(parm[2])
		except: raise Exception("Output filename: <filename>:<outbits>[:f|o|<min>|<min>s[:<max>|<max>s]]")

	return(parm[0],parm[1],None,*vals)

def angle_ab_sym(sym,a,b,c=None,d=None):
	"""Computes the angle of the rotation required to go from Transform A to Transform B under symmetry,
	such that the smallest symmetry-related angle is returned. sym may be either a list of Transforms
	or a symmetry specifier, eg "c4". For the two orientations, specify either
	two Transform objects, or the symmetry followed by four floats in the order 
	AltA,AzA,AltB,AzB. Return in degrees."""
	
	if c!=None :
		A=Transform({"type":"eman","alt":a,"az":b})
		B=Transform({"type":"eman","alt":c,"az":d})
	else :
		A=a
		B=b
	
	# easier to do it here
	Bi=B.inverse()
		
	# needs to be a list of Transforms
	if isinstance(sym,str):
		sym=parsesym(sym).get_syms()
		
	#if not (isinstance(A,Transform) and isinstance(B,Transform)):
		#raise Exception,"angle_ab_sym requries two transforms or 4 angles"

	return min([(A*s*Bi).get_rotation("spin")["omega"] for s in sym])

def kill_process(pid):
	'''
	platform independent way of killing a process
	'''
	import os
	import platform
	platform_string = get_platform()
	if platform_string == "Windows":
		# taken from http://www.python.org/doc/faq/windows/#how-do-i-emulate-os-kill-in-windows
		import win32api
		try:
			# I _think_ this will work but could definitely be wrong
			handle = win32api.OpenProcess(1, 0, pid)
			win32api.TerminateProcess(handle,-1)
			win32api.CloseHandle(handle)
			return 1
		except:
			return 0
	else:
		try:
			os.kill(pid,1)
			return 1
		except:
			return 0

def run(cmd,quiet=False):
	"""shortcut redefined all over the place. Might as well put it in one location."""
	if not quiet: print(cmd)
	return launch_childprocess(cmd)

def launch_childprocess(cmd,handle_err=0):
	'''
	Convenience function to launch child processes
	'''
	p = subprocess.Popen(str(cmd)+" --ppid=%d"%os.getpid(), shell=True)

	if get_platform() == 'Windows':
		error = p.wait()	#os.waitpid does not work on windows
	else:
		error = os.waitpid(p.pid, 0)[1]

	if error and handle_err: 
		print("Error {} running: {}".format(error,cmd))
		sys.exit(1)
		
	return error

def process_running(pid):
	'''
	Platform independent way of checking if a process is running, based on the pid
	'''
	import platform
	import os
	platform_string = get_platform()
	if platform_string == "Windows":
		# taken from http://www.python.org/doc/faq/windows/#how-do-i-emulate-os-kill-in-windows
		import win32api
		try:
			# I _think_ this will work but could definitely be wrong
			handle = win32api.OpenProcess(1, 0, pid)
			#win32api.CloseHandle(handle) # is this necessary?
			return 1
		except:
			return 0
	else:
		try:
			os.kill(pid,0)
			return 1
		except:
			return 0

def memory_stats():
	'''
	Returns [total memory in GB,available memory in GB]
	if any errors occur while trying to retrieve either of these values their retun value is -1
	'''
	import platform
	platform_string = get_platform()
	mem_total = -1
	mem_avail = -1
	if platform_string == "Linux":
		try:
			f = open("/proc/meminfo")
			a = f.readlines()
			mt = a[0].split()
			if mt[0] == "MemTotal:":
				mem_total = old_div(float(mt[1]),1000000.0)
			ma = a[1].split()
			if ma[0] == "MemFree:":
				mem_avail = old_div(float(ma[1]),1000000.0)
		except:
			pass

	elif platform_string == "Darwin":
		import subprocess
		status_total, output_total = subprocess.getstatusoutput("sysctl hw.memsize")
		status_used, output_used = subprocess.getstatusoutput("sysctl hw.usermem")
		total_strings = output_total.split()
		if len(total_strings) >= 2: # try to make it future proof, the output of sysctl will have to be double checked. Let's put it in a unit test in
			total_len = len("hw.memsize")
			if total_strings[0][:total_len] == "hw.memsize":
				try:
					mem_total = old_div(float(total_strings[1]),1000000000.0) # on Mac the output value is in bytes, not kilobytes (as in Linux)
				except:pass # mem_total is just -1

		used_strings = output_used.split()
		mem_used = -1
		if len(used_strings) >=2:
			used_len = len("hw.usermem")
			if used_strings[0][:used_len] == "hw.usermem":
				try:
					mem_used = old_div(float(used_strings[1]),1000000000.0) # on Mac the output value is in bytes, not kilobytes (as in Linux)
				except:pass # mem_used is just -1

		if mem_used != -1 and mem_total != -1:
			mem_avail = mem_total - mem_used

	return [mem_total,mem_avail]

def free_space(p=os.getcwd()):
	'''
	Get the free space on the drive that "p" is on
	return value is in bytes
	Works on Linux - needs checking on other platforms
	'''
	s = os.statvfs(p)
	return s.f_bsize*s.f_bavail

def num_cpus():
	'''
	Returns an estimate of the number of cpus available on the current platform
	'''
	import platform
	platform_string = get_platform()
	if platform_string == "Linux":
		try:
			maxphys=0
			cores=1
			for l in open("/proc/cpuinfo","r"):
				if "physical id" in l: maxphys=max(maxphys,int(l.split(":")[-1]))
				if "cpu cores" in l: cores=int(l.split(":")[-1])
			return cores*(maxphys+1)
		except:
			return 2
	elif platform_string == "Windows":
		try:
			cores = os.getenv("NUMBER_OF_PROCESSORS")
			if cores < 1: return 1 # just for safety
			else: return int(cores)
		except:
			return 2
	elif platform_string == "Darwin":
		import subprocess
		status, output = subprocess.getstatusoutput("sysctl hw.logicalcpu")
		strings = output.split()
		cores = 1 # this has to be true or else it's a really special computer ;)
		if len(strings) >=2:
			used_len = len("hw.logicalcpu")
			if strings[0][:used_len] == "hw.logicalcpu": # this essentially means the system call worked
				try:
					cores = int(strings[1])
				except:pass # mem_used is just -1

		if cores < 1:
			print("warning, the number of cpus was negative (%i), this means the MAC system command (sysctl) has been updated and EMAN2 has not accommodated for this. Returning 1 for the number of cores." %cores)
			cores = 2# just for safety, something could have gone wrong. Maybe we should raise instead
		return cores

	else:
		print("error, in num_cpus - unknown platform string:",platform_string," - returning 2")
		return 2

# returns the local date and time as a string
def local_datetime(secs=-1):
	"""Returns a timestamp as yyyy/mm/dd hh:mm:ss"""
	from time import localtime,strftime
	if secs<0 : secs=time.time()
	t=localtime(secs)
	return strftime("%Y/%m/%d %H:%M:%S",t)

def timestamp_diff(t1,t2):
	"""Takes two timestamps in local_datetime() format and subtracts them, result in seconds.
	difftime() available for convenient display."""
	from time import strptime,mktime

	try:
		tt1=mktime(strptime(t1,"%Y/%m/%d %H:%M:%S"))
		tt2=mktime(strptime(t2,"%Y/%m/%d %H:%M:%S"))
	except:
#		print "time error ",t1,t2
		return 0

	return int(tt2-tt1)

def difftime(secs):
	"""Returns a string representation of a time difference in seconds as a Dd hh:mm:ss style string"""

	d=int(floor(old_div(secs,86400)))
	secs-=d*86400
	h=int(floor(old_div(secs,3600)))
	secs-=h*3600
	m=int(floor(old_div(secs,60)))
	secs=int(secs-m*60)

	if d>0 : return "%dd %2d:%02d:%02d"%(d,h,m,secs)
	if h>0 : return "%2d:%02d:%02d"%(h,m,secs)
	return "   %2d:%02d"%(m,secs)

# returns gm time as a string. For example if it's 11:13 pm on the 18th of June 2008 this will return something like
# '23:13:25.14 18/6/2008'
def gm_time_string():
	'''
	Returns gm time as a string. For example if it's 11:13 pm on the 18th of June 2008 this will return something like '23:13:25.14 18-6-2008'
	The use of '/' is intentionally avoided
	'''

	from time import gmtime,time
	a = time()
	b = gmtime(a)
	astr = str(a)
	idx = str.find(astr,'.')
	decimalseconds = astr[idx:len(astr)]

	val = str(b[3])+':'+str(b[4])+':'+str(b[5])+decimalseconds +' '+str(b[2])+'-'+str(b[1])+'-'+str(b[0])
	return val

def base_name( file_name):
	'''
	Returns a portion of the filename without path or extension/modifiers, and the extension as a tuple
	eg - "test/folder/myfile_test__filtered.hdf:1:10" -> ("myfile_test","hdf")
	'''

	full=os.path.basename(str(file_name))
	return (full.rsplit(".",1)[0].split("__")[0],full.split(":")[0].rsplit(".",1)[-1])

def info_name(file_name):
	"""This will return the name of the info file associated with a given image file, in the form info/basename_info.json"""
	return "info/{}_info.json".format(base_name(file_name)[0])

def display_path(path):
	"""Will generate a suitable reduced path for use in title-bars on windows, etc."""
	
	try: full=os.path.abspath(path)
	except: full=path
	
	full=full.replace("\\","/").split("/")
	if len(full)==1: return full
	return "/".join(full[-3:])

def error_exit(s) :
	"""A quick hack until I can figure out the logging stuff. This function
	should still remain as a shortcut"""
	print(s)
	exit(1)

def warning(s):
	"""prints a warning message as appropriate. In future may support logging, etc."""
	print(s)

def test_image(type=0,size=(128,128)):
	"""Returns a simple standard test image
	type=0  scurve
	type=1	linear increase along x then y
	type=2  square
	type=3  hollow square
"""
	ret=np.zeros(size)
	nx,ny=size

	if type==0 :
		i = np.arange(100)
		centers_x = (nx / 2 + nx / 6.0 * np.sin(i * 2.0 * np.pi / 100.0)).astype(int)
		centers_y = (ny // 4 + i * ny // 200).astype(int)
		xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
		dx = xx[np.newaxis, :, :] - centers_x[:, np.newaxis, np.newaxis]
		dy = yy[np.newaxis, :, :] - centers_y[:, np.newaxis, np.newaxis]		
		dist = np.sqrt(dx**2 + dy**2) * 30.0 / nx
		gaussian = np.exp(-dist**2)
		sinusoid = np.sin(dx * dy) + 0.5
		ret += gaussian * sinusoid
	elif type==1 :
		ret=np.arange(size[0]*size[1]).reshape(size)
	elif type==2:
		ret[nx//4:nx*3//4,ny//4:ny*3//4]=1.0
	elif type==3:
		ret[nx//4:nx*3//4,ny//4:ny*3//4]=1.0
		ret[nx//4+4:nx*3//4-4,ny//4+4:ny*3//4-4]=0.0
	else:
		raise

	return ret
test_image.broken = True

def test_image_3d(type=0,size=(64,64,64)):
	"""Returns a simple standard test image
	type=0  axes, asymmetric (x longest, y shorter, z even shorter) center ->positive
	type=1  linear increase, x then y then z
"""
	ret=np.zeros(size)
	if type==0 :
		ret[nx//2:nx*9//12,ny//2,nz//2]=1.0
		ret[nx//2,ny//2:ny*8//12,nz//2]=1.0
		ret[nx//2,ny//2,nz//2:nz*7//12]=1.0
	elif type==1:
		ret=np.arange(size[0]*size[1]*size[2]).reshape(size)

	return ret

# get a font renderer
def get_3d_font_renderer():
	try:
		from libpyGLUtils2 import EMFTGL,FTGLFontMode
		font_renderer = EMFTGL()
		font_renderer.set_using_display_lists(False)
		font_renderer.set_depth(2)
		font_renderer.set_face_size(12)
		font_renderer.set_font_mode(FTGLFontMode.TEXTURE)
		font_renderer.set_font_file_name(e2getinstalldir()+"/fonts/NotoSansMono-Medium.ttf")
		return font_renderer
	except:
		#traceback.print_exc()
		#print "Unable to import EMFTGL. The FTGL library may not be installed. Text on 3D and some 2D viewers may not work."
		return None

def isosurface(marchingcubes, threshhold, smooth=False):
	"""Return the Isosurface points, triangles, normals(smooth=True), normalsSm(smooth=False)"""
	marchingcubes.set_surface_value(threshhold)
	d = marchingcubes.get_isosurface(smooth)
	return d['points'], d['faces'], d['normals']

# determines if all of the data dimensions are a power of val
# e.g. if ( data_dims_power_of(emdata,2) print "the dimensions of the image are all a power of 2"
def data_dims_power_of(data,val):

	test_cases = [data.get_xsize()]

	if ( data.get_ndim() >= 2 ):
		test_cases.append(data.get_ysize())
	if ( data.get_ndim == 3):
		test_cases.append(data.get_zsize())

	for i in test_cases:
		x = i
		while ( x > 1 and x % val == 0 ): x /= val
		if x != 1:
			return False

	return True

class LSXFile(object):
	"""This class will manage writing entries to LSX files, which are text files with a defined record length for
rapid access. Each line contains an image number, a filename, and an optional comment, referencing a particle
in another actual image file. Files MUST use the Unix /n convention, not the Windows (/r/n) convention.

The file begins with
#LSX
# Whole file comment, one line of any length
# Line length (including \n)
number<\t>filename<\t>comment
...
"""
	def __init__(self,path,ifexists=False, comments="", parent=""):
		"""Initialize the object using the .lst file in 'path'. If 'ifexists' is set, an exception will be raised
if the lst file does not exist. If set, parent specifies the name of another .lst file with identical particles in it,
differing only in metadata. This permits sharing cache files. Do not specify both comments and parent, or parent will be lost"""

		self.path=path
		if len(comments)==0:
			if len(parent)==0:
				comments="# This file is in fast LST format. All lines after the next line have exactly the number of characters shown on the next line. This MUST be preserved if editing."
			else:
				comments=f"# parent={parent} # This file is in fast LST format. All lines after the next line must have identical character count."

		self.ptr=None
		if os.path.isfile(path):
			self.ptr=open(path,"r+")		# file exists
		else:
			if ifexists: raise Exception("Error: lst file {} does not exist".format(path))

			try: os.makedirs(os.path.dirname(path))
			except: pass
			self.ptr=open(path,"w+")	# file doesn't exist
			self.ptr.write("#LSX\n{}\n# 20\n".format(comments))
			self.ptr.flush()

		self.ptr.seek(0)
		l=self.ptr.readline()
		if l==0 or l!="#LSX\n" :
			### UPDATE old LST file to LSX
			if l=="#LST\n" :
				#### This is very similar to rewrite(), but is used to convert LST files to LSX files
				self.seekbase=self.ptr.tell()
				tmpfile=open(self.path+".tmp","w")
				tmpfile.write("#LSX\n{}\n".format(comments))

				# we read the entire file, checking the length of each line
				maxlen=0
				while 1:
					ln=self.ptr.readline().strip()
					if len(ln)==0 : break
					maxlen=max(maxlen,len(ln))

				self.linelen=maxlen+1+4						# we make the lines 8 characters longer than necessary to reduce rewrite calls as "n" gets bigger
				tmpfile.write("# {}\n".format(self.linelen))	# the new line length
				newseekbase=tmpfile.tell()

				fmtstr="{{:<{}}}\n".format(self.linelen-1)	# string for formatting

				self.ptr.seek(self.seekbase)
				while 1:
					ln=self.ptr.readline().strip()
					if len(ln)==0 : break
					tmpfile.write(fmtstr.format(ln))

				# close both files
				tmpfile=None
				self.ptr=None
				self.seekbase=newseekbase

				# rename the temporary file over the original
				os.unlink(self.path)
				os.rename(self.path+".tmp",self.path)
				self.ptr=open(self.path,"r+")
				self.ptr.readline()

			else: raise Exception("ERROR: The file {} is not in #LSX format".format(self.path))
		self.filecomment=self.ptr.readline().strip()
		if "parent=" in self.filecomment: self.parent=self.filecomment.split("parent=")[1].split("#")[0]
		else: self.parent=None

		try: self.linelen=int(self.ptr.readline()[1:])
		except:
			print("ERROR: invalid line length in #LSX file {}".format(self.path))
			raise Exception
		self.seekbase=self.ptr.tell()

		# legacy LST file support
		if self.filecomment.startswith("#keys: "):
			print("WARNING: legacy .lst file containing old-style parameters. Support may be removed in future. Consider rewriting file with e2proclst.py")
			self.filekeys=self.filecomment[7:].split(';')
		else: self.filekeys=None

		# potentially time consuming, but this also gives us self.n
		self.normalize()
		self.lock=threading.Lock()

	def __del__(self):
		self.close()

	def __getitem__(self,n):
		return self.read(n)

	def __setitem__(self,n,tupl):
		if len(tupl)==3 : self.write(n,tupl[0],tupl[1],tupl[2])
		else : self.write(n,tupl[0],tupl[1])

	def __len__(self): return self.n

	def close(self):
		"""Once you call this, you should not try to access this object any more"""
		if self.ptr!=None :
			self.normalize()
			self.ptr=None

	def write(self,n,nextfile,extfile,jsondict=None):
		"""Writes a record to any location in a valid #LSX file.
n : image number in #LSX file, -1 appends, as does n>= current file len
nextfile : the image number in the referenced image file
extfile : the path to the referenced image file (can be relative or absolute, depending on purpose)
jsondict : optional string in JSON format or a JSON compatible dictionary. values will override header values when an image is read.
"""

		self.lock.acquire()
		if jsondict==None : 
			outln="{:d}\t{}".format(nextfile,extfile)
		elif isinstance(jsondict,str) and jsondict[0]=="{" and jsondict[-1]=='}' : 
			outln="{:d}\t{}\t{}".format(nextfile,extfile,jsondict)
		else:
			if not isinstance(jsondict,dict) and jsondict!=None: 
				jsondict={"__default__":jsondict}
			if jsondict!=None:
				jss=json.dumps(jsondict,indent=None,sort_keys=True,separators=(',',':'),default=EMAN3jsondb.obj_to_json)
				outln="{:d}\t{}\t{}".format(nextfile,extfile,jss)
			else: outln="{:d}\t{}".format(nextfile,extfile)
			
		# We can't write in the middle of the file if the existing linelength is too short
		if len(outln)+1>self.linelen : self.rewrite(len(outln)+5)	# +5 just reduces the chance of many rewrites at a small cost in file size

		fmtstr="{{:<{}}}\n".format(self.linelen-1)	# string for formatting
		outln=fmtstr.format(outln)					# padded output line

		if n<0 or n>=self.n :
			self.ptr.seek(0,os.SEEK_END)		# append
			self.n+=1
		else : self.ptr.seek(self.seekbase+self.linelen*n)		# otherwise find the correct location

		self.ptr.write(outln)
		self.lock.release()

	def read(self,n):
		"""Reads the nth record in the file. Note that this does not read the referenced image, which can be
performed with read_image either here or in the EMData class. Returns a tuple (n extfile,extfile,dict). dict
contains decoded information from the stored JSON dictionary. Will also read certain other legacy comments
and translate them into a dictionary."""
		if n>=self.n : raise IndexError("Attempt to read record {} from #LSX {} with {} records".format(n,self.path,self.n))
		self.lock.acquire()
		n=int(n)
		self.ptr.seek(self.seekbase+self.linelen*n)
		ln=self.ptr.readline().strip().split("\t")
		if len(ln)==2 : ln.append("")
		try: ln[0]=int(ln[0])
		except:
			print(f"Error LSXFile.read({n}). {self.seekbase},{self.linelen},{ln}")
			self.lock.release()
			raise(Exception)
		if len(ln[2])<2: ln[2]={}
		else:
			try: ln[2]=json.loads(ln[2],object_hook=EMAN3jsondb.json_to_obj)
			except:
				if self.filekeys!=None:
					vals=ln[2].split(";")
					ln[2]={self.filekeys[i]:eval(vals[i]) for i in range(len(self.filekeys))}
				elif ';Transform' in ln[2]:
					score=float(ln[2].split(";")[0])
					xf=eval(ln[2].split(";")[1])
					ln[2]={"score_align":score,"xform.projection":xf}
				elif ln[2][:9]=="Transform":
					ln[2]={"xform.projection":eval(ln[2])}
				else:
					try:
						dc=eval(ln[2])
						try: score=dc.pop("score")
						except: score=0.0
						ln[2]={"xform.projection":Transform(dc),"score":score}
					except: ln[2]={"lst_comment":ln[2]}

		self.lock.release()
		return ln

	def read_image(self,N,hdronly=False,region=None):
		"""This reads the image referenced by the nth record in the #LSX file. The same task can be accomplished with EMData.read_image,
but this method prevents multiple open/close operations on the #LSX file."""

		raise Exception("Not yet implemented")
		n,fsp,jsondict=self.read(N)
#		print(self.path,n,fsp,jsondict,hdronly,region)
		# ret=EMData()
		# ret.read_image_c(fsp,n,hdronly,region)
		ret["source_path"]=self.path
		ret["source_n"]=N
		if len(jsondict)>0 :
			for k in jsondict: ret[k]=jsondict[k]

		return ret

	def read_into_image(self,ret=None,N=0,hdronly=False,region=None,is_3d=False,imgtype=IMAGE_UNKNOWN):
		"""This reads the image referenced by the nth record in the #LSX file. The same task can be accomplished with EMData.read_image,
but this method prevents multiple open/close operations on the #LSX file."""

		raise Exception("Not yet implemented")
		n,fsp,jsondict=self.read(N)
#		print(self.path,n,fsp,jsondict,hdronly,region)
		# ret.read_image_c(fsp,n,hdronly,region,is_3d,imgtype)
		ret["data_n"]=ret["source_n"]
		ret["data_source"]=ret["source_path"]
		ret["source_path"]=self.path
		ret["source_n"]=N
		if len(jsondict)>0 :
			for k in jsondict: ret[k]=jsondict[k]

	def read_images(self,nlst=None,hdronly=False):
		"""This reads a set of images referenced by the nth record in the #LSX file. This is used by read_images in Python when the file is a LST file
		if nlst is None, the entire file is read."""

		raise Exception("Not yet implemented")
		# organize the images to read by path to take advantage of read_images performance
		# d2r contains tuples (image number in returned array,image number in file (key),extra data dictionary)
		d2r={}
		if nlst is None or len(nlst)==0:
			for i in range(self.n):
				j,p,d=self.read(i)
				try: d2r[p].append((i,j,d,i))
				except: d2r[p]=[(i,j,d,i)]
			ii=self.n
		else:
			# ii is the index of the image in the array we will eventually return, i is the index of the image
			# in the lst file. j is the index in the referenced image file, p. d is the dictionary of 
			# override values from the lst comment field
			for ii,i in enumerate(nlst):
				j,p,d=self.read(int(i))
				try: d2r[p].append((ii,j,d,i))
				except: d2r[p]=[(ii,j,d,i)]
			ii+=1

#		out=open("dbug.txt","w")
		# we read the actual images with calls to read_images for speed
		# then overlay the metadata overrides from the LST file, and put them in the requested read order
		ret=[None]*ii	# ii is left with the total number of images to be returned
		for fsp in d2r:
			tpls=d2r[fsp]
			# imgs=EMData.read_images_c(fsp,[i[1] for i in tpls],IMAGE_UNKNOWN,hdronly)
			for i,tpl in enumerate(tpls):
				imgs[i]["source_path"]=self.path
				try: imgs[i]["source_n"]=int(tpl[3])
				except:
					traceback.print_exc()
					raise Exception(f"Error in read_images: {i},{tpl}")
				for k in tpl[2]: 
					imgs[i][k]=tpl[2][k]
				ret[tpl[0]]=imgs[i]
#				out.write(f"{tpl[0]}\t{i}\t{fsp}\n")

		if None in ret: raise(Exception(f"Error reading {nlst} from {self.path}, {ret.index(None)} is None"))

		return ret

	def normalize(self):
		"""This will read the entire file and insure that the line-length parameter is valid. If it is not,
it will rewrite the file with a valid line-length. """

		self.ptr.seek(self.seekbase)
		self.n=0
		while 1:
			ln=self.ptr.readline()
			if len(ln)==0 :break
			if len(ln)!=self.linelen :
				self.rewrite()
				break
			self.n+=1

	def rewrite(self,minlen=0):
		"""This will reprocess the entire file to make sure the line length is correct. It does take 2 passes,
but should only be used infrequently, so it should be fine. If specified, minlen is the shortest permissible
line length. Used when a line must be added in the middle of the file."""

		self.ptr.seek(0)

		tmpfile=open(self.path+".tmp","w")
		# copy the header lines
		tmpfile.write(self.ptr.readline())
		tmpfile.write(self.ptr.readline())
		self.ptr.readline()

		# we read the entire file, checking the length of each line
		maxlen=minlen
		while 1:
			ln=self.ptr.readline().strip()
			if len(ln)==0 : break
			maxlen=max(maxlen,len(ln))

		self.linelen=maxlen+1+8						# we make the lines 8 characters longer than necessary to reduce rewrite calls as "n" gets bigger
		tmpfile.write("# {}\n".format(self.linelen))	# the new line length
		newseekbase=tmpfile.tell()

		fmtstr="{{:<{}}}\n".format(self.linelen-1)	# string for formatting

		self.ptr.seek(self.seekbase)
		while 1:
			ln=self.ptr.readline().strip()
			if len(ln)==0 : break
			tmpfile.write(fmtstr.format(ln))

		# close both files
		tmpfile=None
		self.ptr=None
		self.seekbase=newseekbase

		# rename the temporary file over the original
		os.unlink(self.path)
		os.rename(self.path+".tmp",self.path)
		self.ptr=open(self.path,"r+")

#		print "rewrite ",self.linelen

def image_eosplit(filename):
	"""This will take an input image stack in LSX or normal image format and produce output .lst (LSX)
files corresponding to even and odd numbered particles. It will return a tuple with two filenames
corresponding to each 1/2 of the data."""

	# create the even and odd data sets
	print("### Creating virtual stacks for even/odd data")
	if open(filename,"r").read(4)=="#LSX" :				# LSX (fast LST) format
		eset=filename.rsplit(".",1)[0]
		oset=eset+"_odd.lst"
		eset+="_even.lst"
		oute=open(eset,"w")
		outo=open(oset,"w")
		inf=open(filename,"r")

		# copy the first 3 lines to both files
		for i in range(3):
			l=inf.readline()
			oute.write(l)
			outo.write(l)

		# then split the rest
		i=0
		while True:
			l=inf.readline()
			if l=="" : break
			if i%2==0 : oute.write(l)
			else: outo.write(l)
			i+=1

		oute=None
		outo=None
	else :								# This means we have a regular image file as an input
		n=file_image_count(filename)
		eset=filename.rsplit(".",1)[0]+"_even.lst"
		oset=filename.rsplit(".",1)[0]+"_odd.lst"

		try : os.unlink(eset)
		except: pass

		oute=LSXFile(eset)
		for i in range(0,n,2): oute.write(-1,i,filename)
		oute=None

		try : os.unlink(oset)
		except: pass

		oute=LSXFile(oset)
		for i in range(1,n,2): oute.write(-1,i,filename)
		oute=None

	return (eset,oset)

def save_lst_params(lst,fsp, overwrite=True):
	"""Saves a LSX file (fsp) with metadata represented by a list of dictionaries (lst).
	each dictionary must contain 'src', the image file containing the actual image and
	'idx' the index in that file. Additional keys will be stored in the LSX metadata
	region. Overwrite existing file by default."""
	if len(lst)==0: raise(Exception,"ERROR: save_lst_params with empty list")
	
	if overwrite:
		if os.path.isfile(fsp):
			os.remove(fsp)
	
	lsx=LSXFile(fsp)
	for d in lst:
		dct=d.copy()
		p=dct.pop("src")
		n=dct.pop("idx")
		lsx.write(-1,n,p,dct)

def load_lst_params(fsp , imgns=None):
	"""Reads the metadata for all of the images in an LSX file (fsp) with an optional list of
	image numbers (imgns, iterable or None)"""
	lsx=LSXFile(fsp,True)
	if imgns==None or len(imgns)==0: imgns=range(lsx.n)
	
	ret=[]
	for i in imgns:
		n,p,d=lsx.read(i)
		ret.append({"idx":n,"src":p,**d})
		
	return ret
	

__doc__ = \
"EMAN classes and routines for image/volume processing in \n\
single particle reconstructions.
"
