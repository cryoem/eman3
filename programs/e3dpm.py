#!/usr/bin/env python
#
# e3dpm.py (EMAN3)
# Steve Ludtke
#
# GUI for Dynamic Point Model (DPM) analysis. Reworked from the EMAN2 e3dpm.py
# (itself a partial conversion of e2gmm.py) to pure Python: EMAN3 + JAX/Flax,
# no C++ dependencies.
#
# A dpm_XX folder is based on a particle stack with per-particle orientations
# (a .lst file) and a points file (Nx4 X,Y,Z,Amp .txt). Training is performed
# by programs.e3dpm_refine (run as a subprocess). The GUI then loads the
# trained decoder and latent vectors, permits exploration of the latent space
# in 2-D, definition of sets, and construction of maps from the sets.
#
import os
import sys
import shutil
import time
import threading
import traceback
import weakref
from queue import Queue

import numpy as np

from EMAN3.EMAN3 import (
	EMArgumentParser, E3init, E3end, E3loadappwin, E3saveappwin,
	js_open_dict, num_path_new, good_size, local_datetime, parse_range,
	error_exit, num_cpus, LSXFile, parsesym
)
from EMAN3.EMAN3jax import Points, Orientations
from EMAN3.io.imageio import ImageIO

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
import pygfx as gfx

from EMAN3.gui.emplot2d import EMPlot2DWidget
from EMAN3.gui.emscene3d import EMScene3DWidget
from EMAN3.gui.valslider import ValSlider

EMANVERSION="e3dpm (EMAN3)"


def copy_lst_with_parent(src,dst):
	"""Copies an LSX .lst file, writing a 'parent=' header comment referencing the source file
	so that StackCache shares the image-data cache created for the original."""
	lsxin=LSXFile(src)
	try:
		lsxout=LSXFile(dst,parent=src)
		for n in range(len(lsxin)):
			idx,extfile,jdict=lsxin.read(n)
			lsxout.write(-1,idx,extfile,jdict)
		lsxout.close()
	finally:
		lsxin.close()


def butval(widg):
	if widg.isChecked(): return 1
	return 0


def showerror(msg,parent=None):
	QtWidgets.QMessageBox.warning(parent,"ERROR",msg)


def good_num(dct):
	"""when passed a dictionary containing numerical (or string numerical) keys, will return the lowest
	"missing" value>=0"""
	if dct is None or len(dct)==0: return 0
	try: ks=set([int(k) for k in dct.keys()])
	except:
		error_exit("dictionary with non-integer keys")

	full=set(range(max(ks)+2))
	return min(full-ks)


# ---------------------------------------------------------------------------
# Reconstruction (pure numpy back-projection with symmetry)
# ---------------------------------------------------------------------------

def backproject(imgs,tytx,mx3d,symmx,box):
	"""Naive back-projection of 2-D projections into a 3-D volume (nearest neighbor).
	imgs - list of (ny,nx) numpy arrays
	tytx - N x 2 array of translations in unit (-0.5,0.5) coordinates (ty,tx order)
	mx3d - 3x3xN orientation matrices (to_mx3d convention, matches projection code)
	symmx - 3x3xM symmetry unit matrices
	box - output box size in pixels (cubic)

	Each image pixel is placed at the 3-D position obtained by inverting the
	projection (pixel = (M_p @ M_s) @ p + tytx) and accumulated with the pixel
	value, for each symmetry unit.
	"""
	N=len(imgs)
	vol=np.zeros((box,box,box),np.float32)
	if N==0 : return vol
	ny,nx=imgs[0].shape

	yy,xx=np.mgrid[0:ny,0:nx]
	U=np.stack((xx.ravel()-nx/2.0,yy.ravel()-ny/2.0,np.zeros(ny*nx)),axis=1)	# (npx,3) centered view-frame pixel coords

	inv=np.linalg.inv
	for i in range(N):
		amp=imgs[i].ravel()
		P3=U-np.array((tytx[i][1],tytx[i][0],0.0))
		for s in range(symmx.shape[2]):
			R=mx3d[:,:,i]@symmx[:,:,s]
			Pref=P3@inv(R).T		# reference frame positions
			ix=(Pref[:,0]+box/2.0).astype(int)
			iy=(Pref[:,1]+box/2.0).astype(int)
			iz=(Pref[:,2]+box/2.0).astype(int)
			m=(ix>=0)&(ix<box)&(iy>=0)&(iy<box)&(iz>=0)&(iz<box)
			if m.any(): np.add.at(vol,(iz[m],iy[m],ix[m]),amp[m])

	return vol


def calc_fsc(vol1,vol2):
	"""Compute the Fourier shell correlation between two volumes, returns an array
	of FSC values binned by Fourier radius in pixels (radius i is bin i)"""
	f1=np.fft.fftn(vol1)
	f2=np.fft.fftn(vol2)
	nz,ny,nx=vol1.shape
	kz=np.fft.fftfreq(nz)*nz
	ky=np.fft.fftfreq(ny)*ny
	kx=np.fft.fftfreq(nx)*nx
	KZ,KY,KX=np.meshgrid(kz,ky,kx,indexing="ij")
	R=np.sqrt(KZ*KZ+KY*KY+KX*KX)
	nshells=min(nz,ny,nx)//2
	fsc=np.zeros(nshells)
	num=np.real(f1*np.conj(f2))
	for i in range(nshells):
		m=(R>=i-0.5)&(R<i+0.5)
		den=np.sqrt(np.sum(np.abs(f1[m])**2)*np.sum(np.abs(f2[m])**2))
		fsc[i]=np.sum(num[m])/den if den>0 else 0
	return fsc


def lowpass_gauss(vol,fc):
	"""Gaussian lowpass filter in Fourier space. fc is the half-amplitude cutoff
	frequency in cycles/pixel"""
	nz,ny,nx=vol.shape
	kz=np.fft.fftfreq(nz)*nz
	ky=np.fft.fftfreq(ny)*ny
	kx=np.fft.fftfreq(nx)*nx
	KZ,KY,KX=np.meshgrid(kz,ky,kx,indexing="ij")
	R=np.sqrt(KZ*KZ+KY*KY+KX*KX)
	mult=np.exp(-2.0*np.log(2.0)*(R/fc)**2)
	return np.fft.ifftn(np.fft.fftn(vol)*mult).real


def downsample2(im):
	"""Factor 2 block-average downsample of a 2-D image"""
	return (im[0::2,0::2]+im[1::2,0::2]+im[0::2,1::2]+im[1::2,1::2])*0.25


def make3d(que,tskn,fsp,imgns,box,apix,sym,fast=False,nset=0):
	"""Thread for 3-D reconstruction in background.
	que - Queue to return result in
	tskn - task number to identify result
	fsp - filename of the particle stack (.lst) with per-particle orientations
	imgns - list of image numbers from fsp to include
	box - reconstruction box size in pixels
	apix - A/pix
	sym - symmetry string, eg "c1", "d2"
	fast - if set, downsample the projections by 2 and reconstruct at half box (then upscale)
	nset - set number, passthrough

	returns in que (tskn,% complete, volume (when 100), resolution, imgns, nset)
	"""

	bigbox=box
	if fast: box=good_size(box//2)

	io=ImageIO(fsp,"r")
	imgs=[]
	tytx=[]
	orts=[]
	n=len(imgns)
	for i,imn in enumerate(imgns):
		if i%500==0: que.put((tskn,10+70*i//max(n,1),None,None,None,None))
		data,hdr=io.read_image(imn)
		if data.ndim==3 : data=data[0]
		data=data.astype(np.float32)
		if fast : data=downsample2(data).astype(np.float32)
		r=hdr["xform.projection"].get_params("spinvec")
		ny=data.shape[0]
		tytx.append((r["ty"]/ny,r["tx"]/ny))
		orts.append((r["v1"],r["v2"],r["v3"]))
	io.close()

	que.put((tskn,82,None,None,None,None))
	orts=np.array(orts,np.float32)
	tytx=np.array(tytx,np.float32)
	mx3d=np.array(Orientations(orts).to_mx3d())	# 3x3xN

	# symmetry units, same convention as the projection loss
	symobj=parsesym(sym)
	symlist=symobj.get_syms()
	symo=Orientations(len(symlist))
	symo.init_from_transforms(symlist)
	symmx=np.array(symo.to_mx3d())	# 3x3xM

	# even/odd halves for FSC
	vole=backproject(imgs[0::2],tytx[0::2],mx3d[:,:,0::2],symmx,box)
	que.put((tskn,90,None,None,None,None))
	volo=backproject(imgs[1::2],tytx[1::2],mx3d[:,:,1::2],symmx,box)
	que.put((tskn,96,None,None,None,None))

	# resolution estimate from gold-standard FSC (0.143 threshold)
	fsc=calc_fsc(vole,volo)
	res=None
	try:
		for r,v in enumerate(fsc[5:]):
			if v<0.143 : res=apix*box/(r+5); break
	except: pass
	if res is None: res=apix*box/max(len(fsc)-1,1)

	ret=(vole+volo).astype(np.float32)
	ret=lowpass_gauss(ret,apix/res)

	if fast:
		# nearest neighbor upsample back to full box
		up=np.repeat(np.repeat(np.repeat(ret,2,axis=0),2,axis=1),2,axis=2)
		ret=up[:bigbox,:bigbox,:bigbox].astype(np.float32)

	que.put((tskn,100,ret,res,list(imgns),nset))


# ---------------------------------------------------------------------------
# Decoder loading (deferred imports keep GUI startup fast)
# ---------------------------------------------------------------------------

def load_decoder(fsp):
	"""Load a decoder trained by programs.e3dpm_refine from a msgpack file"""
	from flax import nnx, serialization
	from programs.e3dpm_refine import Decoder

	with open(fsp,"rb") as f: param=serialization.msgpack_restore(f.read())
	config=param["config"]
	rngs=nnx.Rngs(0)
	model=Decoder(output_dim=config["nout"],latent_dim=config["nlat"],hidden_dims=config["hidden_dims"],activation=config["activation"],rngs=rngs)
	graphdef,state=nnx.split(model)
	state=serialization.from_state_dict(state,nnx.restore_int_paths(param["state"]))
	return nnx.merge(graphdef,state)


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

class TrainThread(QtCore.QThread):
	"""Runs programs.e3dpm_refine as a subprocess. Output goes to the console."""
	finished_ok=Signal(bool)

	def __init__(self,cmd,parent=None):
		super().__init__(parent)
		self.cmd=cmd
		self.ok=False

	def run(self):
		import subprocess
		try:
			print("Running:", " ".join(self.cmd))
			p=subprocess.run(self.cmd,cwd=os.getcwd())
			self.ok=(p.returncode==0)
		except Exception:
			traceback.print_exc()
			self.ok=False
		self.finished_ok.emit(self.ok)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class EMDPM(QtWidgets.QMainWindow):
	"""This is the main window for the e3dpm application"""

	def __init__(self,application,opt):
		"""application is the QApplication instance. opt is the parsed options."""
		QtWidgets.QMainWindow.__init__(self)
		self.options=opt
		self.application=weakref.ref(application)

		self.setWindowTitle("e3dpm (EMAN3)")

		self.dpm=None
		self.jsparm=None
		self.currun=None
		self.currunkey=None
		self.decoder=None
		self.midresult=None
		self.neutralpoints=None
		self.curmaps={}
		self.curmaps_sel={}
		self.cur_dyn_vol=None
		self.line_origin=None
		self.mouseabort=False

		self.threads=[]
		self.threadq=Queue()
		self._train_thread=None

		cen=QtWidgets.QWidget()
		self.setCentralWidget(cen)
		self.gbl=QtWidgets.QGridLayout(cen)
		self.gbl.setColumnStretch(0,0)
		self.gbl.setColumnStretch(1,3)
		self.gbl.setColumnStretch(2,4)

		#### Plot
		self.wplot2d=EMPlot2DWidget()
		self.gbl.addWidget(self.wplot2d,0,1,3,1)
		self.wplot2d.mouseemit=True
		self.wplot2d.set_axis_parms("latent 0","latent 1")

		#### 3-D View
		self.wview3d=EMScene3DWidget()
		self.gbl.addWidget(self.wview3d,0,2,2,1)

		#### Left pane: DPM folder, runs, parameters
		self.gbll=QtWidgets.QGridLayout()
		self.gbl.addLayout(self.gbll,0,0,4,1)

		# dpm_XX folder selection
		self.gblfld=QtWidgets.QGridLayout()
		self.gbll.addLayout(self.gblfld,0,0)

		self.wlistdpm=QtWidgets.QListWidget()
		self.wlistdpm.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Expanding)
		self.update_dpms()
		self.gblfld.addWidget(self.wlistdpm,0,0,1,2)

		self.wbutnewdpm=QtWidgets.QPushButton("New DPM")
		self.wbutnewdpm.setToolTip("Create a new dpm_XX folder based on a particle stack (.lst with orientations) and a points file (.txt, Nx4 X,Y,Z,Amp)")
		self.gblfld.addWidget(self.wbutnewdpm,1,0)

		self.wlpath=QtWidgets.QLabel("-")
		self.wlpath.setToolTip("Source particle stack this DPM is based on")
		self.gblfld.addWidget(self.wlpath,1,1)

		# run selection
		self.gblrun=QtWidgets.QGridLayout()
		self.gbll.addLayout(self.gblrun,1,0)

		self.wlistrun=QtWidgets.QListWidget()
		self.wlistrun.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Expanding)
		self.gblrun.addWidget(self.wlistrun,0,0,1,2)

		self.wbutnewrun=QtWidgets.QPushButton("Create Run")
		self.gblrun.addWidget(self.wbutnewrun,1,0)

		self.wbuttrain=QtWidgets.QPushButton("Train DPM")
		self.wbuttrain.setToolTip("Run programs.e3dpm_refine with the current parameters to train the decoder and latent vectors for this run")
		self.gblrun.addWidget(self.wbuttrain,1,1)

		self.wcbnetstl=QtWidgets.QComboBox()
		for s in ("leaky_5","relu_3","linear"): self.wcbnetstl.addItem(s)
		self.wcbnetstl.setCurrentText("leaky_5")
		self.gblrun.addWidget(self.wcbnetstl,2,0)

		self.wedngauss=QtWidgets.QLabel("0 pts")
		self.wedngauss.setToolTip("Number of points in the model (from the points file)")
		self.gblrun.addWidget(self.wedngauss,2,1)

		#### Parameter form (matches programs.e3dpm_refine options)
		self.gflparm=QtWidgets.QFormLayout()
		self.gbll.addLayout(self.gflparm,2,0)

		self.wedsym=QtWidgets.QLineEdit("c1")
		self.wedsym.setToolTip("Symmetry used in training and reconstruction. cN / dN supported.")
		self.gflparm.addRow("Symmetry:",self.wedsym)

		self.wedmaxres=QtWidgets.QLineEdit("-1")
		self.wedmaxres.setToolTip("Maximum resolution in the loss function (A). -1 = no limit")
		self.gflparm.addRow("Loss Res (A):",self.wedmaxres)

		self.weddim=QtWidgets.QLineEdit("4")
		self.weddim.setToolTip("Number of dimensions in the latent space")
		self.gflparm.addRow("Latent Dim:",self.weddim)

		self.wedtrainiter=QtWidgets.QLineEdit("32")
		self.wedtrainiter.setToolTip("Training iterations per size (total iterations = 2x this)")
		self.gflparm.addRow("Train iter:",self.wedtrainiter)

		self.wedbatch=QtWidgets.QLineEdit("1024")
		self.wedbatch.setToolTip("Particle batch size (less RAM, more epochs)")
		self.gflparm.addRow("Batch size:",self.wedbatch)

		self.wlabruntime=QtWidgets.QLabel("-")
		self.gflparm.addRow("Run:",self.wlabruntime)

		#### Plot controls (below plot)
		self.gblpltctl=QtWidgets.QGridLayout()
		self.gbl.addLayout(self.gblpltctl,3,1)

		self.gblpltctl.addWidget(QtWidgets.QLabel("X Col:",self),0,0,Qt.AlignRight)
		self.wsbxcol=QtWidgets.QSpinBox(self)
		self.wsbxcol.setRange(0,10)
		self.wsbxcol.setValue(0)
		self.gblpltctl.addWidget(self.wsbxcol,0,1,Qt.AlignLeft)

		self.gblpltctl.addWidget(QtWidgets.QLabel("Y Col:",self),1,0,Qt.AlignRight)
		self.wsbycol=QtWidgets.QSpinBox(self)
		self.wsbycol.setRange(0,10)
		self.wsbycol.setValue(1)
		self.gblpltctl.addWidget(self.wsbycol,1,1,Qt.AlignLeft)

		self.wcbpntpln=QtWidgets.QComboBox()
		self.wcbpntpln.addItem("Explore")
		self.wcbpntpln.addItem("Make Set R")
		self.wcbpntpln.addItem("Make Set #")
		self.wcbpntpln.addItem("Line of Sets")
		self.gblpltctl.addWidget(self.wcbpntpln,2,0,1,2)

		self.gblpltctl.addWidget(QtWidgets.QLabel("Rad:",self),3,0,Qt.AlignRight)
		self.wedrad=QtWidgets.QLineEdit("0.2")
		self.wedrad.setToolTip("Radius for including points adjacent to selected point")
		self.gblpltctl.addWidget(self.wedrad,3,1,Qt.AlignRight)

		self.gblpltctl.addWidget(QtWidgets.QLabel("Ptcl:",self),4,0,Qt.AlignRight)
		self.wedcnt=QtWidgets.QLineEdit("1000")
		self.wedcnt.setToolTip("Particles per set in Make Set # / Line of Sets modes")
		self.gblpltctl.addWidget(self.wedcnt,4,1,Qt.AlignRight)

		# clustering / set manipulation buttons
		self.wbutkmeans=QtWidgets.QPushButton("Kmeans")
		self.wbutkmeans.setToolTip("K-means clustering (Sets = number of clusters, Axes = dimensions)")
		self.gblpltctl.addWidget(self.wbutkmeans,0,5)

		self.wbutdbscan=QtWidgets.QPushButton("OpticsDB")
		self.wbutdbscan.setToolTip("OPTICS/DBSCAN clustering (dangerous on large data sets)")
		self.gblpltctl.addWidget(self.wbutdbscan,1,5)

		self.wbuteveryn=QtWidgets.QPushButton("Every N")
		self.wbuteveryn.setToolTip("Every Nth particle in each set (Sets = number of sets)")
		self.gblpltctl.addWidget(self.wbuteveryn,2,5)

		self.wbutlinu=QtWidgets.QPushButton("Line U")
		self.wbutlinu.setToolTip("Split along a single axis uniformly (Sets = number of sets, Axes = axis)")
		self.gblpltctl.addWidget(self.wbutlinu,3,5)

		self.wbutlin=QtWidgets.QPushButton("Line")
		self.wbutlin.setToolTip("Split along a single axis (Sets = number of sets, Axes = axis)")
		self.gblpltctl.addWidget(self.wbutlin,4,5)

		# set buttons
		self.wbutmapnorm=QtWidgets.QPushButton("Build Map")
		self.wbutmapnorm.setToolTip("Reconstruct a full box-size map from the selected sets (even/odd + FSC)")
		self.gblpltctl.addWidget(self.wbutmapnorm,0,6)

		self.wbutmapfast=QtWidgets.QPushButton("Quick Map")
		self.wbutmapfast.setToolTip("Downsampled fast reconstruction of the selected sets")
		self.gblpltctl.addWidget(self.wbutmapfast,1,6)

		self.wbutmapgauss=QtWidgets.QPushButton("Gauss Map")
		self.wbutmapgauss.setToolTip("Build a map from the point model (decoder) of the selected sets")
		self.gblpltctl.addWidget(self.wbutmapgauss,2,6)

		self.wbutsetintsec=QtWidgets.QPushButton("Intersect")
		self.gblpltctl.addWidget(self.wbutsetintsec,0,7)

		self.wbutsetunion=QtWidgets.QPushButton("Union")
		self.gblpltctl.addWidget(self.wbutsetunion,1,7)

		self.wbutsetdel=QtWidgets.QPushButton("Delete")
		self.gblpltctl.addWidget(self.wbutsetdel,2,7)

		self.wbutsetsave=QtWidgets.QPushButton("Save Set")
		self.wbutsetsave.setToolTip("Save selected sets to a new .lst file")
		self.gblpltctl.addWidget(self.wbutsetsave,3,7)

		self.wbutmodelsave=QtWidgets.QPushButton("Save Model")
		self.wbutmodelsave.setToolTip("Save the point model(s) of selected sets to a .txt file")
		self.gblpltctl.addWidget(self.wbutmodelsave,4,7)

		#### 3-D controls (below 3-D view)
		self.gbl3dctl=QtWidgets.QGridLayout()
		self.gbl.addLayout(self.gbl3dctl,2,2,2,1)

		# map list widget
		self.maplist=QtWidgets.QTableWidget()
		self.maplist.setColumnCount(6)
		self.maplist.verticalHeader().hide()
		self.maplist.itemSelectionChanged.connect(self.sel_maptable)
		self.gbl3dctl.addWidget(self.maplist,0,0,5,1)

		self.wbutvtop=QtWidgets.QPushButton("Top")
		self.gbl3dctl.addWidget(self.wbutvtop,0,1)

		self.wbutvbot=QtWidgets.QPushButton("Bot")
		self.gbl3dctl.addWidget(self.wbutvbot,0,2)

		self.wbutvside1=QtWidgets.QPushButton("Side1")
		self.gbl3dctl.addWidget(self.wbutvside1,0,3)

		self.wbutvside2=QtWidgets.QPushButton("Side2")
		self.gbl3dctl.addWidget(self.wbutvside2,0,4)

		self.wbutvoblq=QtWidgets.QPushButton("Oblique")
		self.gbl3dctl.addWidget(self.wbutvoblq,1,2)

		self.wbutvside3=QtWidgets.QPushButton("Side3")
		self.gbl3dctl.addWidget(self.wbutvside3,1,3)

		self.wbutvside4=QtWidgets.QPushButton("Side4")
		self.gbl3dctl.addWidget(self.wbutvside4,1,4)

		# sphere size for point models
		self.wvssphsz=ValSlider(self,(0.2,10.0),"Sphere Size:",1.0)
		self.gbl3dctl.addWidget(self.wvssphsz,2,1,1,4)

		# display toggles
		self.wbutnmdl=QtWidgets.QPushButton("Neutral Model")
		self.wbutnmdl.setCheckable(True)
		self.wbutnmdl.setChecked(True)
		self.wbutnmdl.setToolTip("Show the neutral point model (from the points file)")
		self.gbl3dctl.addWidget(self.wbutnmdl,3,1,1,2)

		self.wbutdmdl=QtWidgets.QPushButton("Dynamic Model")
		self.wbutdmdl.setCheckable(True)
		self.wbutdmdl.setChecked(True)
		self.wbutdmdl.setToolTip("Show the point model decoded from the selected latent vector")
		self.gbl3dctl.addWidget(self.wbutdmdl,3,3,1,2)

		self.wbutdmap=QtWidgets.QPushButton("Dynamic Map")
		self.wbutdmap.setCheckable(True)
		self.wbutdmap.setChecked(True)
		self.wbutdmap.setToolTip("Show the map reconstructed from the selected set")
		self.gbl3dctl.addWidget(self.wbutdmap,4,1,1,2)

		#### 3-D scene nodes
		# neutral point model (from the points file)
		self.neutralplot_node=self.wview3d.add_scatter_plot(name="Neutral Model",data=None,point_size=1.0)
		# dynamic point model (decoded from latent vector)
		self.dynplot_node=self.wview3d.add_scatter_plot(name="Dynamic Model",data=None,point_size=1.0)
		# dynamic map (created lazily when the first map is displayed)
		self.dmap_node=None
		self.dmap_iso=None

		#### Connections
		self.wlistdpm.currentRowChanged.connect(self.sel_dpm)
		self.wbutnewdpm.clicked.connect(self.add_dpm)
		self.wlistrun.currentRowChanged.connect(self.sel_run)
		self.wbutnewrun.clicked.connect(self.new_run)
		self.wbuttrain.clicked.connect(self.do_run_new)
		self.wbutnmap_opt=self.wbutnmdl	# (retained for symmetry with old code)
		self.wbutnmdl.clicked.connect(self.new_3d_opt)
		self.wbutdmdl.clicked.connect(self.new_3d_opt)
		self.wbutdmap.clicked.connect(self.new_3d_opt)
		self.wbutvtop.clicked.connect(self.view_top)
		self.wbutvbot.clicked.connect(self.view_bot)
		self.wbutvoblq.clicked.connect(self.view_oblique)
		self.wbutvside1.clicked.connect(self.view_side1)
		self.wbutvside2.clicked.connect(self.view_side2)
		self.wbutvside3.clicked.connect(self.view_side3)
		self.wbutvside4.clicked.connect(self.view_side4)
		self.wvssphsz.valueChanged.connect(self.new_sph_size)
		self.wbutmapnorm.clicked.connect(self.new_map)
		self.wbutmapfast.clicked.connect(self.new_map_fast)
		self.wbutmapgauss.clicked.connect(self.new_map_gauss)
		self.wbutsetdel.clicked.connect(self.set_del)
		self.wbutsetsave.clicked.connect(self.set_save)
		self.wbutmodelsave.clicked.connect(self.model_save)
		self.wbutsetintsec.clicked.connect(self.intsec_sets)
		self.wbutsetunion.clicked.connect(self.union_sets)
		self.wbutkmeans.clicked.connect(self.do_kmeans)
		self.wbutdbscan.clicked.connect(self.do_dbscan)
		self.wbuteveryn.clicked.connect(self.do_everyn)
		self.wbutlinu.clicked.connect(self.do_linusplit)
		self.wbutlin.clicked.connect(self.do_linesplit)
		self.wsbxcol.valueChanged.connect(self.update_axes_x)
		self.wsbycol.valueChanged.connect(self.update_axes_y)
		self.wplot2d.mousedown.connect(self.plot_mouse)
		self.wplot2d.mouseup.connect(self.plot_mouse_up)
		self.wplot2d.mousedrag.connect(self.plot_mouse_drag)
		self.wplot2d.keypress.connect(self.plot_keyboard)

		E3loadappwin("e3dpm","main",self)

		# select the last (highest-numbered) dpm_XX by default
		if self.wlistdpm.count()>0 : self.wlistdpm.setCurrentRow(self.wlistdpm.count()-1)

		# timer for reconstruction thread results
		self.timer=QtCore.QTimer(self)
		self.timer.timeout.connect(self.timeout)
		self.timer.start(1000)

	def closeEvent(self,event):
		E3saveappwin("e3dpm","main",self)
		event.accept()

	def do_events(self,delay=0.1):
		"""process the event loop with a small delay to allow user abort, etc."""
		t=time.time()
		while (time.time()-t<delay):
			self.application().processEvents()

	# ------------------------------------------------------------------
	# DPM folder management
	# ------------------------------------------------------------------

	def update_dpms(self):
		"""Updates the display of dpm_XX folders"""
		try: dpms=[i for i in sorted(os.listdir(".")) if i[:4]=="dpm_" and os.path.isdir(i)]
		except: dpms=[]
		self.dpms=dpms
		self.wlistdpm.clear()
		for i in self.dpms:
			self.wlistdpm.addItem(i)

	def sel_dpm(self,line):
		"""Called when the user selects a new DPM from the list"""
		if line<0 :
			self.dpm=None
			self.jsparm=None
			self.currun=None
			self.currunkey=None
			self.decoder=None
			self.midresult=None
			self.neutralpoints=None
			self.wlpath.setText("-")
			self.wlistrun.clear()
			self.wplot2d.set_data(None,"map",replace=True)
			self.neutralplot_node.set_data(None)
			self.dynplot_node.set_data(None)
			self.display_dynamic(None)
			self.update_maptable()
			return

		self.dpm=str(self.wlistdpm.item(line).text())
		self.jsparm=js_open_dict(f"{self.dpm}/0_dpm_parms.json")
		self.wlpath.setText(f'Particles: {self.jsparm.getdefault("ptcls","-")}')

		# number of points in the neutral model
		try:
			self.neutralpoints=np.loadtxt(f"{self.dpm}/input_points.txt")
			self.wedngauss.setText(f'{len(self.neutralpoints)} pts')
			box=int(self.jsparm.getdefault("boxsize",128))
			pts=np.column_stack((self.neutralpoints[:,:3]*box,self.neutralpoints[:,3]))
			self.neutralplot_node.set_data(pts)
			# zoom the camera to a reasonable distance for this box size
			ctrl=getattr(self.wview3d,"_controller",None)
			if ctrl is not None:
				ctrl._distance=max(10.0,box*1.2)
				ctrl._update_camera()
				self.wview3d._request_render()
		except:
			self.neutralpoints=None
			self.wedngauss.setText("0 pts")
			print(f"Error reading points file ({self.dpm}/input_points.txt)")

		# run list
		self.wlistrun.clear()
		for k in self.jsparm.keys():
			if k[:4]!="run_" : continue
			self.wlistrun.addItem(k[4:])
		self.currunkey=None

		self.sel_run(self.wlistrun.count()-1)

	def add_dpm(self,clk=False):
		"""Creates a new numbered dpm_XX folder based on a .lst particle stack and a .txt points file"""
		self.application().setOverrideCursor(Qt.BusyCursor)
		try:
			lst=QtWidgets.QFileDialog.getOpenFileName(self,"Please select a particle stack (.lst) with per-particle orientations",os.getcwd(),"LST Files (*.lst);;All Files (*)")[0]
			if lst=="": self.application().restoreOverrideCursor(); return
			lst=os.path.relpath(lst)
			pts=QtWidgets.QFileDialog.getOpenFileName(self,"Please select the corresponding points file (.txt, Nx4 X,Y,Z,Amp)",os.getcwd(),"Text Files (*.txt);;All Files (*)")[0]
			if pts=="": self.application().restoreOverrideCursor(); return
			pts=os.path.relpath(pts)
		except:
			self.application().restoreOverrideCursor()
			return

		try: newdpm=num_path_new("dpm")
		except Exception as e:
			self.application().restoreOverrideCursor()
			showerror(f"Cannot create dpm folder: {e}")
			return

		try:
			lsxout=copy_lst_with_parent(lst,f"{newdpm}/particles.lst")
			shutil.copyfile(pts,f"{newdpm}/input_points.txt")

			io=ImageIO(lst,"r")
			hdr=io.read_header(0)
			io.close()
			boxsize=int(hdr["nx"])
			try: apix=float(hdr["apix_x"])
			except: apix=1.0

			self.jsparm=js_open_dict(f"{newdpm}/0_dpm_parms.json")
			self.jsparm["ptcls"]=lst
			self.jsparm["points"]=pts
			self.jsparm["boxsize"]=boxsize
			self.jsparm["apix"]=apix
			self.jsparm["sym"]="c1"

			self.wlistdpm.addItem(newdpm)
			self.wlistdpm.setCurrentRow(self.wlistdpm.count()-1)
			print(f"Created {newdpm} from {lst} ({boxsize} px, {apix} A/px)")
		except Exception:
			traceback.print_exc()
			try: shutil.rmtree(newdpm)
			except: pass
			showerror("Error setting up new DPM folder, see console for details")
		finally:
			self.application().restoreOverrideCursor()

	# ------------------------------------------------------------------
	# Runs
	# ------------------------------------------------------------------

	def new_run(self,clk=False):
		"""Create a new run"""
		if self.jsparm is None :
			showerror("Please select a DPM first",self)
			return
		name,ok=QtWidgets.QInputDialog.getText(self,"Run Name","Enter a name for the new run. You will still need to train it.")
		if not ok or not name : return
		name=str(name).replace(" ","_")
		if f"run_{name}" not in self.jsparm : self.wlistrun.addItem(name)
		self.currunkey=name
		self.saveparm()		# initialize to avoid messed up defaults later
		self.wlistrun.setCurrentRow(self.wlistrun.count()-1)

	def saveparm(self):
		"""Copies parameters from interface to stored metadata (matches programs.e3dpm_refine options)"""
		if self.jsparm is None or self.currunkey is None : return
		self.currun={}
		self.currun["net_style"]=str(self.wcbnetstl.currentText())
		try: self.currun["maxres"]=float(self.wedmaxres.text())
		except: self.currun["maxres"]=-1
		try: self.currun["ngauss"]=len(self.neutralpoints)
		except: self.currun["ngauss"]=0
		try: self.currun["dim"]=int(self.weddim.text())
		except: self.currun["dim"]=4
		try: self.currun["trainiter"]=int(self.wedtrainiter.text())
		except: self.currun["trainiter"]=32
		try: self.currun["batches"]=int(self.wedbatch.text())
		except: self.currun["batches"]=1024
		self.currun["sym"]=str(self.wedsym.text())
		self.currun["boxsize"]=int(self.jsparm.getdefault("boxsize",128))
		self.currun["apix"]=float(self.jsparm.getdefault("apix",1.0))
		self.currun["time_dynamics"]=local_datetime()
		self.jsparm["run_"+self.currunkey]=self.currun

	def do_run_new(self,clk=False):
		"""Train the DPM for the current run using programs.e3dpm_refine"""
		if self.dpm is None :
			showerror("Select or create a DPM folder first"); return
		if self.currunkey is None :
			showerror("Create a run first"); return
		if os.path.exists(f"{self.dpm}/{self.currunkey}_decoder.msgpack") :
			ans=QtWidgets.QMessageBox.question(self,"Overwrite?",f"Run {self.currunkey} already has a trained model. Retrain and overwrite?")
			if ans==QtWidgets.QMessageBox.No : return

		self.saveparm()
		cur=self.currun
		cmd=[sys.executable,"-m","programs.e3dpm_refine",
			"--points",f"{self.dpm}/input_points.txt",
			"--ptcls",f"{self.dpm}/particles.lst",
			"--sym",cur["sym"],
			"--niter",str(cur["trainiter"]),
			"--nlatent",str(cur["dim"]),
			"--batchsz",str(cur["batches"]),
			"--netstyle",cur["net_style"],
			"--modelout",f"{self.dpm}/{self.currunkey}_decoder.msgpack",
			"--latentout",f"{self.dpm}/{self.currunkey}_mid.txt",
			"--ptclrep",f"{self.dpm}/{self.currunkey}_ptclrep.bin"]
		if cur["maxres"]>0 : cmd+=["--maxres",str(cur["maxres"])]

		self.application().setOverrideCursor(Qt.BusyCursor)
		self.wbuttrain.setEnabled(False)
		self.wbuttrain.setText("Training...")
		print(f"Training run {self.currunkey}... (see console for progress)")
		self._train_thread=TrainThread(cmd,parent=self)
		self._train_thread.finished_ok.connect(self._on_train_done)
		self._train_thread.start()

	def _on_train_done(self,ok):
		self.application().restoreOverrideCursor()
		self.wbuttrain.setEnabled(True)
		self.wbuttrain.setText("Train DPM")
		if ok:
			print(f"Training of run {self.currunkey} complete")
			row=self.wlistrun.currentRow()
			if row<0 and self.wlistrun.count()>0 : row=self.wlistrun.count()-1
			self.wlistrun.setCurrentRow(row)		# triggers sel_run
		else:
			showerror("Training failed, see console for details. If memory exhausted, reduce batch size.")

	def sel_run(self,line):
		"""Called when the user selects a new run from the list. If called with -1 and no run is
		selected, reloads the current run"""
		if line>=0:
			self.currunkey=str(self.wlistrun.item(line).text())
		if self.currunkey is None or self.jsparm is None :
			return

		run=f"run_{self.currunkey}"
		self.currun=self.jsparm.getdefault(run,{"dim":4,"trainiter":32,"batches":1024,"sym":"c1","maxres":-1,"net_style":"leaky_5","time_dynamics":"-"})

		# fill the parameter form
		self.wedsym.setText(str(self.currun.get("sym","c1")))
		self.wedmaxres.setText(str(self.currun.get("maxres",-1)))
		self.weddim.setText(str(self.currun.get("dim",4)))
		self.wedtrainiter.setText(str(self.currun.get("trainiter",32)))
		self.wedbatch.setText(str(self.currun.get("batches",1024)))
		self.wcbnetstl.setCurrentText(self.currun.get("net_style","leaky_5"))
		self.wlabruntime.setText(self.currun.get("time_dynamics","-"))

		# Decoder model for generating points
		self.decoder=None
		try:
			self.decoder=load_decoder(f"{self.dpm}/{self.currunkey}_decoder.msgpack")
		except:
			print(f"Run {self.dpm} -> {self.currunkey}: no trained decoder found yet. Use Train DPM to create one.")

		# Latent representation for every particle
		self.midresult=None
		try:
			if os.path.exists(f"{self.dpm}/{self.currunkey}_aug.txt"):
				self.midresult=np.loadtxt(f"{self.dpm}/{self.currunkey}_aug.txt").transpose()
			elif os.path.exists(f"{self.dpm}/{self.currunkey}_mid.txt"):
				self.augment_mid()
				self.midresult=np.loadtxt(f"{self.dpm}/{self.currunkey}_aug.txt").transpose()
		except:
			traceback.print_exc()
			print(f"Augmented middle layer missing ({self.dpm}/{self.currunkey}_aug.txt)")

		# plot
		if self.midresult is not None:
			self.data=self.midresult
			ncols=len(self.midresult)
			self.wsbxcol.setRange(0,ncols-1)
			self.wsbycol.setRange(0,ncols-1)
			self.wplot2d.set_data(list(self.midresult),"map",replace=True,symsize=2,quiet=True)
			self.wplot2d.axes["map"]=(0,1)
			self.wplot2d.autoscale()
			self.wplot2d.full_refresh()
		else:
			self.data=None
			self.wplot2d.set_data(None,"map",replace=True)

		# list of all data subsets for this run
		allmaps=self.jsparm.getdefault("sets",{})
		try: self.curmaps=dict(allmaps[self.currunkey])
		except: self.curmaps={}
		self.curmaps_sel={}
		for k in self.curmaps:
			m=self.curmaps[k]
			try: m[5]=np.array(m[5])
			except: pass

		self.update_maptable()

		# clear the dynamic displays
		self.dynplot_node.set_data(None)
		self.display_dynamic(None)
		self.new_3d_opt()

	def augment_mid(self):
		"""Take the latent layer from the network result and create the "augmented" file with 2-D PCA
		in the first 2 columns (the idea is that these columns can be later expanded with other
		dimensionality reduction methods)"""
		try:
			if not os.path.exists(f"{self.dpm}/{self.currunkey}_mid.txt"): raise Exception
			mid=np.loadtxt(f"{self.dpm}/{self.currunkey}_mid.txt")		# (Nptcl, nlatent)

			copycol=True
			adc=None
			try:
				from sklearn.decomposition import PCA
				adc=PCA(n_components=2).fit_transform(mid).transpose()	# (2,Nptcl)
				copycol=False
			except:
				print("PCA not available, copying latent columns")

			out=open(f"{self.dpm}/{self.currunkey}_aug.txt","w")
			out.write("# PCA0; PCA1; "+"Latent; "*mid.shape[1]+"\n")
			for i in range(mid.shape[0]):
				if copycol:
					v=mid[i,0] if mid.shape[1]>=1 else 0.0
					w=mid[i,1] if mid.shape[1]>=2 else v
					out.write(f"{v:1.4f}\t{w:1.4f}\t")
				else:
					out.write(f"{adc[0][i]:1.4f}\t{adc[1][i]:1.4f}\t")
				out.write("\t".join([f"{r:1.4f}" for r in mid[i]])+"\n")
			out.close()
		except:
			traceback.print_exc()
			print(f"Middle layer missing ({self.dpm}/{self.currunkey}_mid.txt)")

	# ------------------------------------------------------------------
	# Plot interactions
	# ------------------------------------------------------------------

	def _plot_shape_node(self,name,verts,color=(1.0,1.0,0.0,1.0)):
		"""Add a simple line/point overlay to the plot (world coords)"""
		w=self.wplot2d
		if not hasattr(w,"_get_or_create_node") : return
		try:
			node=gfx.Points(gfx.Geometry(positions=np.array(verts,dtype=np.float32)),
				gfx.PointsMaterial(color=color,size=3.0))
			node.render_order=998
			w._get_or_create_node(name,node)
			w._dirty=True
			w._request_render()
		except Exception:
			pass

	def _plot_circle(self,cx,cy,rad):
		a=np.linspace(0,2*np.pi,64)
		verts=list((cx+rad*np.cos(a[i]),cy+rad*np.sin(a[i]),0.05) for i in range(len(a)))
		self._plot_shape_node("region",verts)

	def _plot_line(self,p0,p1):
		self._plot_shape_node("genline",[p0[0],p0[1],0.05,p1[0],p1[1],0.05])

	def _plot_clear(self):
		w=self.wplot2d
		if hasattr(w,"_clear_shapes"):
			w._clear_shapes()
			w._dirty=True
			w._request_render()

	def plot_keyboard(self,event):
		"""keyboard events from the 2-D plot"""
		if event.key()==Qt.Key_Escape : self.mouseabort=True

	def plot_mouse(self,event,loc):
		"""mouse down on the 2-D plot"""
		if self.midresult is None : return
		self.mouseabort=False
		self.line_origin=None
		mmode=str(self.wcbpntpln.currentText())
		xcol=min(self.wsbxcol.value(),self.midresult.shape[0]-1)
		ycol=min(self.wsbycol.value(),self.midresult.shape[0]-1)

		try: rad=float(self.wedrad.text())
		except:
			print("invalid radius, using 0.05")
			rad=0.05

		if mmode=="Explore":
			# This will produce a list of indices where the distance in the plane is less than the specified rad
			ptdist=((self.midresult[xcol]-loc[0])**2+(self.midresult[ycol]-loc[1])**2<(rad**2)).nonzero()[0]
			nptcl=len(ptdist)

			# no particles in the circle!
			if nptcl==0 : return

			self._plot_circle(loc[0],loc[1],rad)
			self._show_set_model(ptdist)
			return
		elif mmode=="Line of Sets":
			self.line_origin=loc
		elif mmode in ("Make Set R","Make Set #"):
			return
		else : print("mode error")

	def plot_mouse_drag(self,event,loc):
		mmode=str(self.wcbpntpln.currentText())
		if self.mouseabort:
			print("Abort")
			self._plot_clear()
			self.mouseabort=False
			self.line_origin=None
			return

		if mmode=="Line of Sets" and self.line_origin is not None:
			self._plot_line(self.line_origin,loc)

	def plot_mouse_up(self,event,loc):
		if self.midresult is None : return
		mmode=str(self.wcbpntpln.currentText())
		dim=self.currun.get("dim",4)
		xcol=self.wsbxcol.value()
		ycol=self.wsbycol.value()

		if self.mouseabort:
			print("Abort")
			self._plot_clear()
			self.mouseabort=False
			self.line_origin=None
			return

		try: rad=float(self.wedrad.text())
		except:
			print("invalid radius, using 0.05")
			rad=0.05
		try: ptperset=int(self.wedcnt.text())
		except:
			print("invalid particle count, using 4000")
			ptperset=4000

		self._plot_clear()

		if mmode=="Make Set R":
			# This will produce a list of indices where the distance in the plane is less than the specified rad
			ptdist=((self.midresult[xcol]-loc[0])**2+(self.midresult[ycol]-loc[1])**2<(rad**2)).nonzero()[0]
			if len(ptdist)==0 : return

			vec=np.zeros(len(self.midresult))
			vec[xcol]=loc[0]
			vec[ycol]=loc[1]
			nset=good_num(self.curmaps)
			newmap=[None,local_datetime(),list(vec),0,0,ptdist]
			self.curmaps[str(nset)]=newmap
			self.sets_changed(nset)
			return
		if mmode=="Make Set #":
			ptdist=((self.midresult[xcol]-loc[0])**2+(self.midresult[ycol]-loc[1])**2)
			sel=np.argsort(ptdist)[:ptperset]

			vec=np.zeros(len(self.midresult))
			vec[xcol]=loc[0]
			vec[ycol]=loc[1]
			nset=good_num(self.curmaps)
			newmap=[None,local_datetime(),list(vec),0,0,sel]
			self.curmaps[str(nset)]=newmap
			self.sets_changed(nset)
			return
		elif mmode=="Line of Sets":
			if self.line_origin is None : return
			import math
			dst=math.hypot((self.line_origin[0]-loc[0]),(self.line_origin[1]-loc[1]))
			if dst<1e-9 : return
			stps=int(dst/rad)+1		# number of points to generate along the line

			# now we generate a set for each point on the line
			nset=good_num(self.curmaps)
			for i in range(stps+1):
				cen=((self.line_origin[0]*(stps-i)+loc[0]*i)/stps,(self.line_origin[1]*(stps-i)+loc[1]*i)/stps)
				ptdist=((self.midresult[xcol]-cen[0])**2+(self.midresult[ycol]-cen[1])**2)
				sel=np.argsort(ptdist)[:ptperset]

				vec=np.zeros(len(self.midresult))
				vec[xcol]=loc[0]
				vec[ycol]=loc[1]
				newmap=[None,local_datetime(),list(vec),0,0,sel]
				self.curmaps[str(nset+i)]=newmap
			self.sets_changed(nset+stps)
			return

	def _show_set_model(self,ptdist):
		"""Run the mean latent vector of the selected points through the decoder and display the
		decoded point model in the 3-D view"""
		if self.decoder is None :
			showerror("Train the DPM first (no decoder available for this run)")
			return
		dim=self.currun.get("dim",4)
		import jax.numpy as jnp
		latent=np.mean(self.midresult[2:2+dim][:,ptdist],0,keepdims=True)		# the mean latent vector over selected points
		gauss=np.array(self.decoder(jnp.array(latent))).reshape(-1,4)
		box=int(self.jsparm.getdefault("boxsize",128))
		pts=np.column_stack((gauss[:,:3]*box,gauss[:,3]))
		self.dynplot_node.set_data(pts)
		self.wbutdmdl.setChecked(True)
		self.new_3d_opt()

	# ------------------------------------------------------------------
	# Sets
	# ------------------------------------------------------------------

	def update_maptable(self,sel=None):
		"""Updates the table containing all of the current computed dynamic maps"""
		self.maplist.clear()
		self.maplist.setHorizontalHeaderLabels(["Set","Ptcl","Map","Size","Res","Datestamp"])
		self.maplist.setRowCount(len(self.curmaps))
		for i,k in enumerate(sorted([int(i) for i in self.curmaps])):
			m=self.curmaps[str(k)]

			twi=QtWidgets.QTableWidgetItem(f"{k}")
			twi.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
			self.maplist.setItem(i,0,twi)

			twi=QtWidgets.QTableWidgetItem(f"{len(m[5])}")
			twi.setFlags(Qt.ItemIsEnabled)
			self.maplist.setItem(i,1,twi)

			twi=QtWidgets.QTableWidgetItem(f"{m[0]}" if m[0] is not None else "-")
			twi.setFlags(Qt.ItemIsEnabled)
			self.maplist.setItem(i,2,twi)

			twi=QtWidgets.QTableWidgetItem(f"{m[3]}")
			twi.setFlags(Qt.ItemIsEnabled)
			self.maplist.setItem(i,3,twi)

			twi=QtWidgets.QTableWidgetItem(f"{m[4]:1.1f}" if m[4] else "-")
			twi.setFlags(Qt.ItemIsEnabled)
			self.maplist.setItem(i,4,twi)

			twi=QtWidgets.QTableWidgetItem(f"{m[1]}")
			twi.setFlags(Qt.ItemIsEnabled)
			self.maplist.setItem(i,5,twi)

		self.maplist.resizeColumnsToContents()
		self.curmaps_sel={}
		if not sel is None and sel>=0 and sel<self.maplist.rowCount():
			self.maplist.selectRow(sel)

	def sel_maptable(self):
		"""When items are selected in the list of generated maps"""
		if self.midresult is None : return
		ss=max(3,8-int(np.log10(max(len(self.midresult[0]),2))))
		if len(self.maplist.selectedItems())>0:
			try:
				# ensure the full data set is present
				if "map" not in self.wplot2d.data:
					self.wplot2d.set_data(list(self.data),"map",symsize=2,replace=False,quiet=True)
				for i in self.maplist.selectedItems():
					key=i.text()
					smap=self.curmaps[key]
					self.curmaps_sel[key]=smap
					if not isinstance(smap[5],np.ndarray) : smap[5]=np.array(smap[5])
					self.wplot2d.set_data(list(self.data[:,smap[5]]),f"set_{int(key):02d}",symsize=ss,quiet=True)
			except:
				print("Error displaying selected points, e3dpm.py:sel_maptable()")
				return
		else:
			# clear any set overlays
			for key in [k for k in self.wplot2d.data if k.startswith("set_")]:
				self.wplot2d.clear_data(key)
			self.curmaps_sel={}

		self.wplot2d.autoscale()
		self.wplot2d.full_refresh()

		# when a single set is selected, we display the corresponding model
		if len(self.maplist.selectedItems())==1:
			key=self.maplist.selectedItems()[0].text()
			smap=self.curmaps[key]
			ptdist=smap[5]
			if len(ptdist)>1000: ptdist=ptdist[::len(ptdist)//1000]
			self._show_set_model(ptdist)

			# display the reconstructed map if we have one
			if smap[0] is not None:
				try:
					io=ImageIO(f"{self.dpm}/set_maps.hdf","r")
					vol,_=io.read_image(smap[0])
					io.close()
					if vol.ndim==3 : vol=vol[0]
					self.display_dynamic(vol)
				except:
					print("Error: map missing for ",smap)
			else:
				self.display_dynamic(None)
		else:
			self.display_dynamic(None)

	def sets_changed(self,nnew=None):
		"""Update the map table and persist the sets to the dpm parameters file"""
		curmaps={}
		for k in self.curmaps:
			curmaps[k]=list(self.curmaps[k][:5])+[list(self.curmaps[k][5])]
		allmaps=self.jsparm.getdefault("sets",{})
		allmaps[self.currunkey]=curmaps
		self.jsparm["sets"]=allmaps
		self.update_maptable(nnew)

	def intsec_sets(self):
		if len(self.curmaps_sel)==0:
			showerror("Select one or more sets in the table"); return
		sets=[]
		for key in self.curmaps_sel:
			sets.append(set(self.curmaps_sel[key][5]))
		nset=good_num(self.curmaps)
		sel=np.array(list(set.intersection(*sets)))
		vec=np.zeros(len(self.midresult))
		newmap=[None,local_datetime(),list(vec),0,0,sel]
		self.curmaps[str(nset)]=newmap
		self.sets_changed(nset)

	def union_sets(self):
		if len(self.curmaps_sel)==0:
			showerror("Select one or more sets in the table"); return
		sets=[]
		for key in self.curmaps_sel:
			sets.append(set(self.curmaps_sel[key][5]))
		nset=good_num(self.curmaps)
		sel=np.array(list(set.union(*sets)))
		vec=np.zeros(len(self.midresult))
		newmap=[None,local_datetime(),list(vec),0,0,sel]
		self.curmaps[str(nset)]=newmap
		self.sets_changed(nset)

	def set_del(self,ign=None):
		"""Delete an existing set (only if a map hasn't been computed for it)"""
		if len(self.maplist.selectedItems())==0: return
		delerr=False
		for i in self.maplist.selectedItems():
			k=i.text()
			if k not in self.curmaps : continue
			st=self.curmaps[k]
			if st[0] is not None:
				delerr=True
				continue
			del self.curmaps[k]

		self.sets_changed()

		if delerr: QtWidgets.QMessageBox.warning(None, "Warning", "Warning: sets with computed maps not deleted!")

	def set_save(self,ign=None):
		"""Save one or more sets to a new LST file for further processing"""
		if len(self.maplist.selectedItems())==0:
			showerror("One or more sets must be selected. Selected sets will be merged into a single .lst file.")
			return

		imgns=[]
		ks=[]
		for i in self.maplist.selectedItems():
			k=i.text()
			if k not in self.curmaps : continue
			ks.append(k)
			imgns.extend(list(self.curmaps[k][5]))
		imgns=sorted(imgns)

		lsin=LSXFile(f"{self.dpm}/particles.lst")
		outfsp=f"{self.dpm}/{self.currunkey}_set_{'-'.join(ks)}.lst"
		try: os.unlink(outfsp)
		except: pass
		lsout=LSXFile(outfsp)
		for n in imgns:
			lsout[-1]=lsin[n]		# despite the odd notation, -1 does work this way
		lsin.close()
		lsout.close()

		print(f"{len(imgns)} selected particles saved to {outfsp}")

	def model_save(self,ign=None):
		"""Save the point model of selected sets to a new .txt file"""
		if len(self.maplist.selectedItems())==0:
			showerror("One or more sets must be selected. Points from selected sets will be saved (mean model per set, merged).")
			return
		if self.decoder is None:
			showerror("Train the DPM first (no decoder available for this run)")
			return
		dim=self.currun.get("dim",4)
		import jax.numpy as jnp
		gausses=[]
		for i in self.maplist.selectedItems():
			key=i.text()
			if key not in self.curmaps : continue
			smap=self.curmaps[key]
			ptdist=smap[5]
			if len(ptdist)>1000: ptdist=ptdist[::len(ptdist)//1000]		# We don't really need every point to get a pretty good average position for the set
			latent=np.mean(self.midresult[2:2+dim][:,ptdist],0,keepdims=True)		# the mean latent vector over selected points
			gausses.append(np.array(self.decoder(jnp.array(latent))).reshape(-1,4))

		if len(gausses)==0 : return
		gauss=gausses[0] if len(gausses)==1 else np.vstack(gausses)

		outfsp=f"{self.dpm}/{self.currunkey}_gauss_{'-'.join([i.text() for i in self.maplist.selectedItems()])}.txt"
		try: os.unlink(outfsp)
		except: pass

		np.savetxt(outfsp,gauss,delimiter="\t")

		print(f"{gauss.shape} points saved to {outfsp}")

	# ------------------------------------------------------------------
	# Clustering / set generation
	# ------------------------------------------------------------------

	def do_kmeans(self):
		print("kmeans ...")
		self.application().setOverrideCursor(Qt.BusyCursor)
		try: nseg=int(self.wedcnt.text().split(",")[0])
		except: nseg=4
		try: nseg=int(QtWidgets.QInputDialog.getText(self,"Sets","Number of clusters:")[0] or nseg)
		except: pass
		cols=np.array([self.wsbxcol.value(),self.wsbycol.value()])
		try: nset=max([int(k) for k in self.curmaps])+1
		except: nset=0

		try:
			from sklearn.cluster import KMeans
			kmseg=KMeans(n_clusters=nseg,init='k-means++')
			classes=kmseg.fit_predict(self.data[cols].transpose())
		except:
			self.application().restoreOverrideCursor()
			showerror("Problem with K-means parameters (sklearn required). Axes must be specified with X Col/Y Col.")
			return

		for i in range(nseg):
			ptdist=np.where(classes==i)[0]
			newmap=[None,local_datetime(),(list(cols),list(kmseg.cluster_centers_[i])),0,0,ptdist]
			self.curmaps[str(nset+i)]=newmap

		self.sets_changed()
		print("done")
		self.application().restoreOverrideCursor()

	def do_everyn(self):
		print("Every N in a group ...")
		self.application().setOverrideCursor(Qt.BusyCursor)
		try: nseg=int(QtWidgets.QInputDialog.getText(self,"Sets","Number of sets:")[0])
		except:
			self.application().restoreOverrideCursor()
			showerror("N Sets must be an integer for this operation")
			return

		try: nset=max([int(k) for k in self.curmaps])+1
		except: nset=0

		for i in range(nseg):
			newmap=[None,local_datetime(),([0],i),0,0,np.arange(i,self.data.shape[1],nseg)]
			self.curmaps[str(nset+i)]=newmap

		self.sets_changed()
		print("done")
		self.application().restoreOverrideCursor()

	def do_dbscan(self):
		self.application().setOverrideCursor(Qt.BusyCursor)
		print("OpticsDB ...")
		t0=time.time()
		try:
			from sklearn.cluster import OPTICS
		except:
			self.application().restoreOverrideCursor()
			showerror("OPTICS not available (sklearn required)")
			return
		cols=np.array([self.wsbxcol.value(),self.wsbycol.value()])

		try: nset=max([int(k) for k in self.curmaps])+1
		except: nset=0

		try:
			opseg=OPTICS(n_jobs=-1,cluster_method="dbscan",max_eps=0.25)
			classes=opseg.fit_predict(self.data[cols].transpose())
			nseg=max(classes)+1
			if (nseg>100):
				print(f"{nseg} classes found, aborting")
				self.application().restoreOverrideCursor()
				return
		except:
			traceback.print_exc()
			self.application().restoreOverrideCursor()
			return

		for i in range(nseg):
			ptdist=np.where(classes==i)[0]
			try:
				sel=self.data[:,ptdist]
				cen=np.mean(sel[cols],1)
				newmap=[None,local_datetime(),[list(cols),list(cen)],0,0,ptdist]
			except:
				newmap=[None,local_datetime(),(list(cols),None),0,0,ptdist]
			self.curmaps[str(nset+i)]=newmap

		self.sets_changed()
		print(f"done ({time.time()-t0}s)")
		self.application().restoreOverrideCursor()

	def do_linesplit(self):
		self.application().setOverrideCursor(Qt.BusyCursor)
		print("Split along a single axis ...")
		try: nseg=int(QtWidgets.QInputDialog.getText(self,"Sets","Number of sets:")[0])
		except:
			self.application().restoreOverrideCursor()
			showerror("For Line mode please specify a single number for Sets (number of sets)")
			return
		col=self.wsbycol.value()

		try: nset=max([int(k) for k in self.curmaps])+1		# starting set number
		except: nset=0

		cen=np.mean(self.data[col])
		minv=max(cen-np.std(self.data[col])*2.5,np.min(self.data[col]))	# mean-standard deviation*2.5 or actual minimum value
		maxv=min(cen+np.std(self.data[col])*2.5,np.max(self.data[col])) # mean+standard deviation*2.5 or actual maximum value

		classes=np.floor(nseg*(self.data[col]-minv)/(maxv-minv)).astype("int32") # particles outside the minv->maxv range will be excluded from the sets

		for i in range(nseg):
			ptdist=np.where(classes==i)[0]
			try:
				cen=np.mean(self.data[col,ptdist])
				newmap=[None,local_datetime(),[col,cen],0,0,ptdist]
			except:
				newmap=[None,local_datetime(),(col,None),0,0,ptdist]
			self.curmaps[str(nset+i)]=newmap

		self.sets_changed()
		self.application().restoreOverrideCursor()

	def do_linusplit(self):
		self.application().setOverrideCursor(Qt.BusyCursor)
		print("Split along a single axis (uniformly) ...")
		try: nseg=int(QtWidgets.QInputDialog.getText(self,"Sets","Number of sets:")[0])
		except:
			self.application().restoreOverrideCursor()
			showerror("For Line U mode please specify a single number for Sets (number of sets)")
			return
		col=self.wsbycol.value()

		try: nset=max([int(k) for k in self.curmaps])+1		# starting set number
		except: nset=0

		idxs=np.argsort(self.data[col])
		grpn=len(idxs)//nseg
		for i in range(nseg):
			newmap=[None,local_datetime(),(col,None),0,0,idxs[i*grpn:(i+1)*grpn]]
			self.curmaps[str(nset+i)]=newmap

		self.sets_changed()
		self.application().restoreOverrideCursor()

	# ------------------------------------------------------------------
	# Maps
	# ------------------------------------------------------------------

	def new_map(self,ign=None):
		"""Start threads to generate a new full scale map for all selected sets"""
		if len(self.maplist.selectedItems())==0:
			showerror("Select one or more sets in the table"); return
		box=good_size(int(self.jsparm.getdefault("boxsize",128))*5//4)
		for i in self.maplist.selectedItems():
			k=i.text()
			if k not in self.curmaps : continue
			st=self.curmaps[k]
			self.curmaps_sel[k]=st
			self.threads.append(threading.Thread(target=make3d,args=(self.threadq,len(self.threads),f"{self.dpm}/particles.lst",list(st[5]),box,self.jsparm.getdefault("apix",1.0),self.currun.get("sym","c1"),False,k)))
			self.threads[-1].start()
			print(f"Thread {len(self.threads)-1} started with {len(st[5])} particles")

	def new_map_fast(self,ign=None):
		"""Start threads to generate a new downsampled map for all selected sets"""
		if len(self.maplist.selectedItems())==0:
			showerror("Select one or more sets in the table"); return
		box=good_size(int(self.jsparm.getdefault("boxsize",128))*5//4)
		for i in self.maplist.selectedItems():
			k=i.text()
			if k not in self.curmaps : continue
			st=self.curmaps[k]
			self.curmaps_sel[k]=st
			self.threads.append(threading.Thread(target=make3d,args=(self.threadq,len(self.threads),f"{self.dpm}/particles.lst",list(st[5]),box,self.jsparm.getdefault("apix",1.0),self.currun.get("sym","c1"),True,k)))
			self.threads[-1].start()
			print(f"Thread {len(self.threads)-1} started (fast) with {len(st[5])} particles")

	def new_map_gauss(self,ign=None):
		"""generate point maps from set centers (decoder point model)"""
		if len(self.maplist.selectedItems())==0:
			showerror("Select one or more sets in the table"); return
		if self.decoder is None:
			showerror("Train the DPM first (no decoder available for this run)")
			return
		dim=self.currun.get("dim",4)
		box=int(self.jsparm.getdefault("boxsize",128))
		apix=float(self.jsparm.getdefault("apix",1.0))
		import jax.numpy as jnp

		try:
			io=ImageIO(f"{self.dpm}/set_maps.hdf","r"); nmaps=io.nimg; io.close()
		except: nmaps=0

		io=ImageIO(f"{self.dpm}/set_maps.hdf","rw")
		last=None
		for i in self.maplist.selectedItems():
			k=i.text()
			if k not in self.curmaps : continue
			smap=self.curmaps[k]

			ptdist=smap[5]
			if len(ptdist)>1000: ptdist=ptdist[::len(ptdist)//1000]		# We don't really need every point to get a pretty good average position for the set
			latent=np.mean(self.midresult[2:2+dim][:,ptdist],0,keepdims=True)		# the mean latent vector over selected points
			gauss=np.array(self.decoder(jnp.array(latent))).reshape(-1,4)

			# build a volume from the point model
			vol=np.array(Points(gauss).volume_np(box)).astype(np.float32)

			io.write_image(nmaps,vol,{"nx":box,"ny":box,"nz":box,"apix_x":apix,"apix_y":apix,"apix_z":apix})

			# store metadata to identify maps
			# set # (key), 0) map #, 1) timestamp, 2) center, 3) map size, 4) est resolution, 5) list of points
			newmap=[nmaps,local_datetime(),None,box,-1,list(smap[5])]
			self.curmaps[k]=newmap
			last=vol
			nmaps+=1
		io.close()

		self.sets_changed()
		if last is not None: self.display_dynamic(last)

	def timeout(self):
		"""Handles the results of completed reconstruction threads"""
		while not self.threadq.empty():
			ret=self.threadq.get()
			tskn,prog,vol,res,imgns,nset=ret
			if prog==100:
				print(f"Thread {tskn} complete, resolution {res:1.1f} A")
				bs=int(self.jsparm.getdefault("boxsize",128))

				# clip to the particle box size
				ps=(vol.shape[0]-bs)//2
				if ps>0: vol=vol[ps:ps+bs,ps:ps+bs,ps:ps+bs]

				# normalize for display
				mean=vol.mean()
				std=vol.std()
				if std>0: vol=((vol-mean)/std).astype(np.float32)

				# store the reconstructed map
				try:
					io=ImageIO(f"{self.dpm}/set_maps.hdf","r"); nmaps=io.nimg; io.close()
				except: nmaps=0
				apix=float(self.jsparm.getdefault("apix",1.0))
				io=ImageIO(f"{self.dpm}/set_maps.hdf","rw")
				io.write_image(nmaps,vol,{"nx":bs,"ny":bs,"nz":bs,"apix_x":apix,"apix_y":apix,"apix_z":apix})
				io.close()

				# store metadata to identify maps
				newmap=[nmaps,local_datetime(),None,bs,res,list(imgns)]
				self.curmaps[str(nset)]=newmap
				self.sets_changed()

				self.display_dynamic(vol)

				# clean up the thread
				t=self.threads[tskn]
				t.join()
				self.threads[tskn]=None		# just gets longer and longer...
			else:
				print(f"Thread {tskn} at {prog}%")

	def display_dynamic(self,vol):
		"""Displays a new dynamic map, used multiple places so condensed here"""
		self.cur_dyn_vol=vol
		if vol is None:
			if self.dmap_node is not None:
				self.dmap_node.visible=False
				if self.dmap_iso is not None: self.dmap_iso.visible=False
				self.wview3d._request_render()
			return

		if self.dmap_node is None:
			self.dmap_node=self.wview3d.add_data(name="Dynamic Map",data=vol.astype(np.float32))
			self.dmap_iso=self.wview3d.add_isosurface(name="Isosurface",threshold=0.5,color=(1.0,0.4,0.2),parent=self.dmap_node)
		else:
			self.dmap_node._data=vol.astype(np.float32)
			self.dmap_node._rebuild_bbox()
			if self.dmap_iso is not None: self.dmap_iso._rebuild_volume()

		self.dmap_node.visible=True
		if self.dmap_iso is not None: self.dmap_iso.visible=True
		self.wview3d._request_render()

	# ------------------------------------------------------------------
	# 3-D view
	# ------------------------------------------------------------------

	def _set_view(self,az,alt,phi):
		ctrl=getattr(self.wview3d,"_controller",None)
		if ctrl is None : return
		ctrl._azimuth=az
		ctrl._altitude=alt
		ctrl._phi=phi
		ctrl._update_camera()
		self.wview3d._request_render()

	def view_top(self,tmp=None):
		self._set_view(0,0,0)

	def view_bot(self,tmp=None):
		self._set_view(180,0,0)

	def view_oblique(self,tmp=None):
		self._set_view(45,22.5,0)

	def view_side1(self,tmp=None):
		self._set_view(0,90,0)

	def view_side2(self,tmp=None):
		self._set_view(90,90,0)

	def view_side3(self,tmp=None):
		self._set_view(22.5,90,0)

	def view_side4(self,tmp=None):
		self._set_view(45,90,0)

	def new_sph_size(self,newval=1.0):
		self.neutralplot_node.set_point_size(self.wvssphsz.value)
		self.dynplot_node.set_point_size(self.wvssphsz.value)
		self.wview3d._request_render()

	def new_3d_opt(self,clk=False):
		"""When the user changes selections for the 3-D display"""
		self.neutralplot_node.visible=self.wbutnmdl.isChecked()
		self.dynplot_node.visible=self.wbutdmdl.isChecked()
		if self.dmap_node is not None:
			self.dmap_node.visible=self.wbutdmap.isChecked()
			if self.dmap_iso is not None: self.dmap_iso.visible=self.wbutdmap.isChecked()
		self.wview3d._request_render()

	def set3dvis(self,neumdl=1,dynmdl=1,blankplot=0):
		"""sets the visibility of various 3-D display components. 1 enables, 0 disables, -1 leaves unchanged"""
		if neumdl>=0: self.wbutnmdl.setChecked(neumdl>0)
		if dynmdl>=0: self.wbutdmdl.setChecked(dynmdl>0)
		self.new_3d_opt()

	# ------------------------------------------------------------------
	# Plot axis selection
	# ------------------------------------------------------------------

	def _set_plot_axis(self,which,col):
		w=self.wplot2d
		if "map" not in w.axes : return
		xa,ya=w.axes["map"]
		if which=="x" : w.axes["map"]=(col,ya)
		else : w.axes["map"]=(xa,col)
		w.autoscale()
		w.full_refresh()

	def update_axes_x(self,x):
		self._set_plot_axis("x",x)

	def update_axes_y(self,y):
		self._set_plot_axis("y",y)


def main():
	progname=os.path.basename(sys.argv[0])
	usage="""e3dpm [options]

	This program is designed to work interactively with point mixture models, automating as much
	as possible of the tasks associated with studying dynamics using DPMs. It requires a particle
	stack with known orientations (.lst) and a points file (.txt) as a starting point."""

	parser=EMArgumentParser(usage=usage,version=EMANVERSION)
	parser.add_argument("--threads",default=-1,type=int,help="Number of alignment threads to run in parallel on a single computer.")
	parser.add_argument("--ppid",type=int,help="Set the PID of the parent process, used for cross platform PPID",default=-1)

	global options
	(options, args) = parser.parse_args()

	if options.threads<1 : options.threads=num_cpus()

	app=QtWidgets.QApplication(sys.argv)
	emdpm=EMDPM(app,options)
	emdpm.show()
	emdpm.raise_()
	sys.exit(app.exec())


if __name__=="__main__":
	main()
