#!/usr/bin/env python
#
# e3browsercache.py (EMAN3)
# Steve Ludtke
#
# Standalone updater for the e3display file-browser metadata columns. Scans a
# folder and writes a browser metadata cache (a .json file stored in the same
# location as the .lst file caches) that the browser widget reads to populate
# its Type/Dim/N/Size columns. Because this runs as a separate process, slow
# I/O and per-file header reads never block the browser UI: the browser keeps a
# live list of files and refreshes its metadata columns whenever this program
# writes a newer cache (the widget polls the cache file's timestamp).
#
import os
import re
import sys
import time

from EMAN3.EMAN3 import EMArgumentParser, browser_cache_write, browser_cache_read, BROWSER_CACHE_VERSION, local_datetime
from EMAN3.io.imageio import ImageIO

EMANVERSION="e3browsercache (EMAN3)"

# File extensions ImageIO can open as image data. If present, the file is
# probed for its image count and dimensions.
IMAGE_EXTS={
	".jpg",".jpeg",".pgm",".png",
	".tif",".tiff",".mrcs",".mrsc",".raw",".hed",".img",".hdf",".h5",
	".ico",".icos",".mrc",".hdr",".omap",".dns6",".brix",".am",".amira",".vtk",
	".pif",".eer",".spi",".lst"
}

# Friendly "Type" names for non-image files, keyed by extension.
NONIMAGE_TYPE={
	".lst":"LSX",
	".json":"JSON",
	".txt":"Text",
	".log":"Text",
	".csv":"Data",
	".tsv":"Data",
	".dat":"Data",
	".pdb":"PDB",
	".bdf":"PDB",
	".html":"HTML",
	".htm":"HTML",
	".pdf":"PDF"
}

# Type strings the browser knows how to act on (used to validate cached entries).
KNOWN_TYPES={"Image","Image Stack","LSX","Data","File","Folder"} | set(NONIMAGE_TYPE.values())


def probe_image(full):
	"""Try to open 'full' as image data. Returns (type, dim, nimg) on success, None otherwise."""
	try:
		io=ImageIO(full,"r")
		nimg=io.nimg
		if nimg and nimg>0:
			hdr=io.read_header(0)
			nx=int(hdr.get("nx",0)); ny=int(hdr.get("ny",0)); nz=int(hdr.get("nz",0))
			if ny==1: dim=str(nx)
			elif nz==1: dim="%d x %d"%(nx,ny)
			else: dim="%d x %d x %d"%(nx,ny,nz)
			ftype="Image" if nimg==1 else "Image Stack"
			io.close()
			return ftype,dim,int(nimg)
		io.close()
	except Exception:
		pass
	return None


def is_text(full):
	"""Cheap text-ness test: readable as text, no NUL bytes in the first 64KB."""
	try:
		with open(full,"rb") as f:
			chunk=f.read(65536)
		return b"\x00" not in chunk
	except Exception:
		return False


def count_nonempty_lines(full):
	"""Count non-empty lines, streamed (memory-safe for large files)."""
	try:
		n=0
		with open(full,"rb") as f:
			tail=b""
			while True:
				chunk=f.read(1<<16)
				if not chunk: break
				lines=(tail+chunk).splitlines(keepends=True)
				if lines and not lines[-1].endswith(b"\n"):
					tail=lines.pop()
				else:
					tail=b""
				for raw in lines:
					if raw.strip(): n+=1
			if tail.strip(): n+=1
		return n
	except Exception:
		return 0


_SPLITS=re.compile(r"[\s,;]+")

def plot_parse_shape(full):
	"""Try to parse 'full' exactly the way EMPlot2DWidget/EMPlot3DWidget do:
	skip blank lines and lines starting with '#', strip '#' comments, split fields on
	whitespace/comma/semicolon, every field must be a float, and every data line
	must have the same field count. The first data line may be a non-numeric
	header row (column labels). Returns (ncol, nrows) on success, None otherwise.
	"""
	try:
		ncol=None
		nrows=0
		with open(full,"r") as f:
			for line in f:
				if len(line)<2 or line[0]=="#": continue
				fields=[x for x in _SPLITS.split(line.split("#")[0]) if x]
				if not fields: continue
				if ncol is None:
					try:
						for x in fields: float(x)
						nrows+=1
					except ValueError:
						pass		# first line is a header (column labels)
					ncol=len(fields)
					continue
				if len(fields)!=ncol:
					return None
				for x in fields: float(x)
				nrows+=1
		return (ncol,nrows) if (ncol and nrows) else None
	except Exception:
		return None


def classify(full):
	"""Return a metadata dict {type, dim, nimg} for a single non-directory file."""
	ext=os.path.splitext(full)[1].lower()
	if ext in IMAGE_EXTS:
		r=probe_image(full)
		if r is not None:
			t,dim,nimg=r
			return {"type":t,"dim":dim,"nimg":nimg}
	# Any text file (regardless of extension) that the 2D/3D plot widgets can
	# parse is reported as Data with its Ncol x Nrow shape.
	r=plot_parse_shape(full)
	if r is not None:
		ncol,nrows=r
		return {"type":"Data","dim":"%d x %d"%(ncol,nrows),"nimg":None}
	if is_text(full):
		typ=NONIMAGE_TYPE.get(ext,"Text")
		return {"type":typ,"dim":"%d ln"%count_nonempty_lines(full),"nimg":None}
	if ext in NONIMAGE_TYPE:
		return {"type":NONIMAGE_TYPE[ext],"dim":None,"nimg":None}
	if ext=="":
		return {"type":"File","dim":None,"nimg":None}
	return {"type":ext.lstrip(".").upper(),"dim":None,"nimg":None}


def scan_folder(folder,hidedot=False,verbose=0):
	"""Scan one folder (non-recursive) and return a dict of filename -> metadata.
	Incremental: if an existing cache of the current version exists, entries whose
	file modification time is unchanged are reused as-is; only new or modified
	files are re-classified (which keeps probe/header/line-count checks affordable
	on large folders). Caches from an older version are re-classified fully.
	"""
	entries={}
	_ts,old_entries,old_path,old_version=browser_cache_read(folder)
	reuse=(old_version==BROWSER_CACHE_VERSION)
	try:
		names=os.listdir(folder)
	except Exception as e:
		print("Cannot list %s: %s"%(folder,e))
		return entries
	for name in sorted(names):
		if hidedot and name[0]==".": continue
		full=os.path.join(folder,name)
		e={}
		st=None
		try:
			st=os.stat(full)
			e["size"]=int(st.st_size)
		except Exception:
			pass
		try:
			if os.path.isdir(full):
				e["type"]="Folder"
			else:
				reused=False
				o=None
				if reuse and st is not None:
					# Reuse a cached entry only if the file is unchanged and the
					# cached classification is one the browser can act on (this
					# re-classifies stale entries produced by older versions).
					o=old_entries.get(name,{})
					cached_type=o.get("type")
					if (o.get("mtime")==st.st_mtime and cached_type is not None
							and cached_type in KNOWN_TYPES):
						reused=True
				if reused:
					entries[name]=dict(o)
					if verbose>0:
						print("%-40s %-14s %-18s %s  (cached)"%(name,str(o.get("type")),str(o.get("dim")),str(o.get("nimg"))))
					continue
				e.update(classify(full))
				if st is not None: e["mtime"]=st.st_mtime
		except Exception:
			e.setdefault("type","Unknown")
		entries[name]=e
		if verbose>0:
			print("%-40s %-14s %-18s %s"%(name,str(e.get("type")),str(e.get("dim")),str(e.get("nimg"))))
	return entries


def main():

	usage="""e3browsercache.py [folder]

Scan a folder and write a browser metadata cache (a .json file stored in the
same location as the .lst file caches) that the e3display file browser uses to
populate its Type/Dim/N/Size columns.

Because it runs as a separate process, slow I/O and per-file header reads never
block the browser UI. The browser keeps a live list of files and refreshes its
metadata columns whenever it detects a newer cache file (by timestamp).

The cache location is controlled the same way as the .lst caches:
EMAN3_CACHE_PATH env variable, then project info/project.json, then ./cache.
"""
	parser=EMArgumentParser(usage=usage,version=EMANVERSION)
	parser.add_argument("--hidedot",action="store_true",help="skip hidden files (names beginning with '.')")
	parser.add_argument("--verbose","-v",action="store_true",help="print one line per scanned file")

	(options,args)=parser.parse_args()
	folder=args[0] if len(args)>0 else "."

	if not os.path.isdir(folder):
		print("Error: %s is not a directory"%folder)
		sys.exit(1)

	t0=time.time()
	entries=scan_folder(folder,hidedot=options.hidedot,verbose=1 if options.verbose else 0)
	path=browser_cache_write(folder,entries)

	ndir=sum(1 for e in entries.values() if e.get("type")=="Folder")
	nimg=sum(1 for e in entries.values() if e.get("type") in ("Image","Image Stack"))
	ntxt=len(entries)-ndir-nimg
	print("%s: cached %d entries (%d folder, %d image, %d other) for %s in %.2fs"%(
		local_datetime(),len(entries),ndir,nimg,ntxt,folder,time.time()-t0))
	print("Cache file: %s"%path)


if __name__=="__main__":
	main()
