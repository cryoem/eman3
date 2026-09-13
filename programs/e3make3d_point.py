#!/usr/bin/env python
#
# e3make3d_point.py (EMAN3)
# Steve Ludtke
#
# Point-based 3-D reconstruction (ported from the EMAN2 e3make3d_point.py to pure
# Python: EMAN3 modules + JAX/optax, no C++ dependencies).
#
import sys
import os
import math
import time
import traceback
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import optax

from EMAN3.EMAN3 import EMArgumentParser, E3init, E3end, parsesym, error_exit, local_datetime, good_size, LSXFile
from EMAN3.EMAN3jax import (
	StackCache, Points, Orientations, EMStack2D, EMStack3D,
	rad_img_int, rad2_img, rad_vol_int, to_jax,
	jax_fsc_jit, prj_frcs, sym_prj_frcs_loss,
	point_gradient_step_optax, point_gradient_step_ctf_optax, point_gradient_step_layered_ctf_optax,
	point_project_simple_sym_fn, point_project_ctf_sym_fn, point_project_layered_ctf_sym_fn
)
from EMAN3.io.imageio import ImageIO

try: os.mkdir(".jaxcache")
except: pass

# We cache the JIT compilation results to speed up future runs
jax.config.update("jax_compilation_cache_dir", "./.jaxcache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

jax.config.update("jax_default_matmul_precision", "float32")

EMANVERSION="e3make3d_point (EMAN3)"


def lowpass_gauss(vol,fc):
	"""Gaussian lowpass filter in Fourier space. fc is the half-amplitude cutoff in cycles/pixel"""
	nz,ny,nx=vol.shape
	kz=np.fft.fftfreq(nz)*nz
	ky=np.fft.fftfreq(ny)*ny
	kx=np.fft.fftfreq(nx)*nx
	KZ,KY,KX=np.meshgrid(kz,ky,kx,indexing="ij")
	R=np.sqrt(KZ*KZ+KY*KY+KX*KX)
	mult=np.exp(-2.0*np.log(2.0)*(R/fc)**2)
	return np.fft.ifftn(np.fft.fftn(vol)*mult).real


def normalize_edgemean(vol):
	"""Linear transform such that the mean of the 3-pixel border around the edges is 0 and the std is 1
	(same semantics as the EMAN2 "normalize.edgemean" processor)"""
	ny,nx=vol.shape[0],vol.shape[1]
	b=3
	edgemean=(vol[0:b,:,:].mean()+vol[ny-b:,:,:].mean()+vol[:,0:b,:].mean()+vol[:,ny-b:,:].mean()+vol[:,:,0:b].mean()+vol[:,:,nx-b:].mean())/6.0
	std=vol.std()
	if std<1e-12 : return vol-edgemean
	return (vol-edgemean)/std


def apply_symmetry(vol,sym):
	"""Average a real-space (z,y,x) volume over the symmetry units of sym (pure rotation about the center,
	trilinear interpolation). Same intent as the EMAN2 "xform.applysym" processor applied to a map"""
	orts=Orientations()
	orts.init_from_transforms(sym.get_syms())
	mxs=np.array(orts.to_mx3d())				# (3,3,M)
	nz,ny,nx=vol.shape
	c=np.array(((nz-1)/2.0,(ny-1)/2.0,(nx-1)/2.0))
	z,y,x=np.meshgrid(np.arange(nz),np.arange(ny),np.arange(nx),indexing="ij")
	P=np.stack((x,y,z),axis=-1).reshape((-1,3))		# (n,3) xyz order
	shp=np.array(vol.shape)
	out=np.zeros_like(vol)
	cnt=np.zeros(P.shape[0])				# per-voxel count of symmetry units that contributed
	for i in range(mxs.shape[2]):
		q=(P-c)@mxs[:,:,i].T+c			# where to sample in the input volume
		valid=(q>=0).all(axis=1)&(q<shp).all(axis=1)
		if not valid.any(): continue
		qv=q[valid]
		i0=np.floor(qv).astype(np.int32)
		frac=qv-i0
		i1=np.minimum(i0+1,shp-1)					# clamp the +1 tap to the last index
		def g(ix,iy,iz): return vol[iz,iy,ix]		# vol is (z,y,x); coords are xyz
		vv=(
			g(i0[:,0],i0[:,1],i0[:,2])*(1-frac[:,0])*(1-frac[:,1])*(1-frac[:,2])+\
			g(i0[:,0],i0[:,1],i1[:,2])*(1-frac[:,0])*(1-frac[:,1])*frac[:,2]+\
			g(i0[:,0],i1[:,1],i0[:,2])*(1-frac[:,0])*frac[:,1]*(1-frac[:,2])+\
			g(i0[:,0],i1[:,1],i1[:,2])*(1-frac[:,0])*frac[:,1]*frac[:,2]+\
			g(i1[:,0],i0[:,1],i0[:,2])*frac[:,0]*(1-frac[:,1])*(1-frac[:,2])+\
			g(i1[:,0],i0[:,1],i1[:,2])*frac[:,0]*(1-frac[:,1])*frac[:,2]+\
			g(i1[:,0],i1[:,1],i0[:,2])*frac[:,0]*frac[:,1]*(1-frac[:,2])+\
			g(i1[:,0],i1[:,1],i1[:,2])*frac[:,0]*frac[:,1]*frac[:,2]
		)
		out.reshape(-1)[valid]+=vv
		cnt[valid]+=1.0
	m=cnt>0
	out.reshape(-1)[m]/=cnt[m]
	return out


def write_volume(path,vol,hdr=None):
	"""Append a real-space (z,y,x) volume to an image file via ImageIO"""
	nz,ny,nx=vol.shape
	if hdr is None: hdr={}
	hdr=dict(hdr)
	hdr.setdefault("nx",nx); hdr.setdefault("ny",ny); hdr.setdefault("nz",nz)
	io=ImageIO(path,"rw")
	io.write_image(-1,np.asarray(vol,dtype=np.float32),hdr)
	io.close()


def compute_fsc(vol_a,vol_b):
	"""FSC curve between two real-space volumes of identical shape. Returns (1, nr) numpy FSC array.
	The volumes must be cubes and rad_vol_int(cube_size) must have been called first"""
	fa=jnp.fft.rfftn(to_jax(np.asarray(vol_a)),axes=(0,1,2))
	fb=jnp.fft.rfftn(to_jax(np.asarray(vol_b)),axes=(0,1,2))
	return np.array(jax_fsc_jit(fa[jnp.newaxis,...],fb))


def main():

	usage="""e3make3d_point.py <projections.lst>


	"""
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	parser.add_argument("--volout", type=str,help="Volume output file. Note that volumes will be appended to an existing file", default="threed.hdf")
	parser.add_argument("--pointout", type=str,help="Point list output file",default=None)
	parser.add_argument("--volfiltlp", type=float, help="Lowpass filter to apply to output volume in A, 0 disables, default=5", default=5)
	parser.add_argument("--frc_weight", type=float, help="Testing only at present", default=-1)
	parser.add_argument("--apix", type=float, help="A/pix override for raw data", default=-1)
	parser.add_argument("--thickness", type=float, help="For tomographic data specify the Z thickness in A to limit the reconstruction domain", default=-1)
	parser.add_argument("--outbox",type=int,help="output boxsize, permitting over/undersampling (impacts A/pix)", default=-1)
	parser.add_argument("--postclip",type=int,help="Trim the output volumes to the specified (square) box size in pixels (no impact on A/pix)", default=-1)
	parser.add_argument("--initpoint",type=int,help="Points in the first pass, scaled with stage, default=500", default=500)
	parser.add_argument("--savesteps", action="store_true",help="Save the point parameters for each refinement step, for debugging and demos")
	parser.add_argument("--combineiters", type=int, help="Specify an additional number of iterations to add to the end of refinement, volume will use all Point positions during these iterations", default=-1)
	parser.add_argument("--spt", action="store_true",help="subtomogram averaging mode, changes optimization steps")
	parser.add_argument("--quick", action="store_true",help="single particle mode with less thorough refinement, but faster results")
	parser.add_argument("--ctf", type=int,help="0=no ctf, 1=single ctf, 2=layered ctf",default=0)
	parser.add_argument("--keep", type=str, help="The fraction of images to use, based on quality scores (1.0 = use all). Optionally 3 values for SPT only: 3d qual, 2d qual, other",default="1.0")
	parser.add_argument("--ptcl3d_id", type=str, help="only use 2-D particles with matching ptcl3d_id parameter (lst file/header, use : for range with excluded upper limit)",default=None)
	parser.add_argument("--class", dest="classid", type=int, help="only use 2-D particles with matching class parameter (lst file/header)",default=-1)
	parser.add_argument("--sym", type=str,help="symmetry if no value is given then the model is assumed to have no symmetry.\nChoices are: i, c, d, tet, icos, or oct", default="c1")
	parser.add_argument("--fscdebug", type=str,help="Compute the FSC of the final map with a reference volume for debugging",default=None)
	parser.add_argument("--gpudev",type=int,help="GPU Device, default 0", default=0)
	parser.add_argument("--gpuram",type=int,help="Maximum GPU ram to allocate in MB, default=4096", default=4096)
	parser.add_argument("--profile", action="store_true",help="Used for code development only, not routine use")
	parser.add_argument("--cachepath",type=str,help="folder for storing the cached images, should be on a high speed drive, M.2 if possible. Default='./cache/'",default=".")
	parser.add_argument("--score",type=str,help="If set, will generate the specified .lst file with score set and plottable scores.txt",default=None)
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9], higher number means higher level of verbosity")

	(options, args) = parser.parse_args()
	llo=E3init(sys.argv,options.ppid)

	os.putenv("EMAN3_CACHE_PATH",options.cachepath)
	options.keep=[float(k) for k in options.keep.split(',')]
	if len(options.keep)<3: options.keep=[options.keep[0]]*3

	if len(args)==0 or args[0][-4:]!=".lst" : error_exit("Only LST input files are supported. If working with an HDF with embedded parameters, create a .lst file and extract the parameters with e2proclst.py")
	lsx=LSXFile(args[0])
	nptcl=len(lsx)

	# Particle selection based on various options
	if min(options.keep)==1.0 and options.ptcl3d_id is None: selimg=set(range(nptcl))
	else:
		# in spt mode we consider all 3 keep values, looking at scores on 3-D and 2-D particles, The third value was used for something else in the original program. Here we just combine with the second
		if options.spt :
			p3d=defaultdict(list)		# construct dictionary keyed by 3d particle, with scores and image numbers in value
			p2ds=[]						# construct list of (score,#,ptcl3d_id) for 2-D particles at the same time
			for i,l in enumerate(lsx):
				p3d[l[2]["ptcl3d_id"]].append((l[2]["score"],i))
				p2ds.append((l[2]["score"],i,l[2]["ptcl3d_id"]))

			p3ds=[(min(v),k) for k,v in p3d.items()]	# list of (score,3d_id) for 3d particles, we are using the best score among the 2-D particles, _not_ the average
			p3ds.sort()
			if options.verbose>2:
				out=open("dbg_3d.txt","w")
				for s,i in p3ds: out.write(f"{i}\t{s[0]}\t{s[1]}\n")
				out=None
			good_3d=set([i for s,i in p3ds[:int(len(p3ds)*options.keep[0])]])	# set of good ptcl3d_id numbers
			# include only particles in the specified range (if specified)
			if options.ptcl3d_id is not None:
				try:
					idmin=int(options.ptcl3d_id)
					idmax=idmin+1
				except:
					idmin,idmax=[int(i) for i in options.ptcl3d_id.split(":")]
				mrg=set(range(idmin,idmax))
				good_3d=good_3d.intersection(mrg)
			if options.verbose>0: print(f"SPT mode: {len(good_3d)}/{len(p3d)} 3-D particles selected")

			# 2D filtering
			p2ds.sort()
			if options.verbose>2:
				out=open("dbg_2d.txt","w")
				for s,i,t in p2ds: out.write(f"{i}\t{s}\t{t}\n")
				out=None
			selimg=set([i for s,i,t in p2ds[:int(len(p2ds)*options.keep[1]*options.keep[2])] if t in good_3d])

		# normal mode (not spt) where we just have a score per-particle
		else:
			p2ds=[]						# construct list of (score,#) for 2-D particles at the same time
			for i,l in enumerate(lsx):
				p2ds.append((l[2]["score"],i))
			p2ds.sort()
			selimg=set([i for s,i in p2ds[:int(len(p2ds)*options.keep[0])]])

	if options.classid>=0:
		selcls=set([i for i in range(len(lsx)) if lsx[i][2]["class"]==options.classid])
		selimg=selimg.intersection(selcls)

	selimg=np.array(list(selimg))   # need to convert set to list before going to array or we wind up with an array with a set in it
	selimg.sort()
	if options.quick : selimg=selimg[:8192+4096]
	nptcl=len(selimg)
	if options.verbose>0: print(f"{nptcl}/{len(lsx)} 2-D images selected")

	if options.profile:
		selimg=np.array(range(0,min(2050,nptcl)))
		nptcl=len(selimg)
		print("WARNING: profiling mode enabled. Actual point map results will not be useful, used for development only!")

	# input data dimensions and A/pix (read first image header via ImageIO)
	io=ImageIO(args[0],"r")
	firsthdr=io.read_header(0)
	io.close()
	nxraw=int(firsthdr["nx"])
	if options.apix>0: apix=options.apix
	else:
		try: apix=float(firsthdr["apix_x"])
		except: apix=1.0
	if options.thickness>0: zmax=options.thickness/(apix*nxraw*2.0)		# instead of +- 0.5 Z range, +- zmax range
	else: zmax=0.5
	if options.outbox>0: outsz=options.outbox
	else: outsz=min(1024,nxraw)

	if options.verbose: print(f"Input data box size {nxraw}x{nxraw} at {apix} A/pix. Maximum downsampled size for refinement {nxraw}. Thickness limit +-{zmax}. {nptcl} input images")

	if options.savesteps:
		try: os.unlink("steps.hdf")
		except: pass

	refdata=None
	if options.fscdebug is not None:
		io=ImageIO(options.fscdebug,"r")
		refdata,_=io.read_image(0)
		io.close()
		refdata=np.asarray(refdata)
		# critical: jax_fsc_jit requires the radius image for this cube size to be precomputed
		rad_vol_int(refdata.shape[0])
	else: dbugvol=None
	dbugvol=EMStack3D(refdata).do_fft().jax if refdata is not None else None

	if options.verbose: print(f"{nptcl} particles at {nxraw}^3")

	# definition of downsampling sequence for stages of refinement
	# 0) #ptcl, 1) downsample, 2) iter, 3) frc weight, 4) amp threshold, 5) replicate, 6) repl. spread, 7) step coef (no longer used)
	# replication skipped in final stage
	if  options.spt:
		stages=[
			[2**12,32,  32,1.8, -1,1,.05, 3.0],
			[2**12,32,  32,1.8, -1,2,.05, 1.0],
			[2**13,64,  48,1.5, -2,2,.04,1.0],
			[2**13,64,  48,1.5, -2,8,.02,0.5],
			[2**15,128, 32,1.2, -3,16,.01,2],
			[2**17,256, 16,1.2, -2,32,.01,3],
			[2**18,512, 8,1.2, -2,64,.01,3],
			[2**18,1024, 8,1.2, -2,0,.01,3]
		]
	elif options.quick or options.profile:
		stages=[
			[512,   16,24,1.8,-3  ,1,.01, 2.0],
			[512,   16,24,1.8, 0  ,4,.01, 1.0],
			[1024,  32,24,1.5, 0  ,8,.005,1.5],
			[4096,  64,24,1.2,-1.5,16,.003,1.0],
			[8192, 256,24,1.0,-2  ,0,.003,1.0],
		]
	else:
		stages=[
			[512,   16,32,1.8,-3  ,1,.01, 2.0],
			[512,   16,32,1.8, 0  ,4,.01, 1.0],
			[1024,  32,32,1.5, 0  ,4,.005,1.5],
			[1024,  32,32,1.5,-0.5  ,8,.005,1.0],
			[4096,  64,24,1.2,-1,8,.003,1.0],
			[8192,  128,24,1.2,-1.5,8,.003,1.0],
			[16384, 256,32,1.0,-2 ,16,.003,1.0],
			[65536*2, 512,32,1.0,-2 ,0,.001,0.75]
		]

	# limit sampling to (at most) the box size of the raw data
	# we do this by altering stages to help with jit compilation
	for i in range(len(stages)):
		stages[i][1]=min(stages[i][1],nxraw)
		if options.frc_weight>0: stages[i][3]=options.frc_weight

	# Setting up symmetry
	sym=parsesym(options.sym)
	sym_orts = Orientations()
	sym_orts.init_from_transforms(sym.get_syms())
	symmx = sym_orts.to_mx3d()


	# prior to jax 0.7.x there was a sharding problem in JAX causing a crash related to
	# each image in the batch getting turned into a separate argument when JIT compiling
	# in 0.7.x, larger batches can be used and have some advantages
	if int(jax.__version__.split(".")[1])>6 :
		if options.spt: batchsize=640
		else: batchsize=1024//len(sym_orts)
	else:
		if options.spt: batchsize=320
		else: batchsize=192

	if options.combineiters>0:
		refineiters=5
		stages[-1][2]+=(options.combineiters-1)*refineiters # Increase the number of iterations so can save Points
		final_point = Points()
		final_point._data = []
	times=[time.time()]

	# Cache initialization
	if options.verbose: print(f"{local_datetime()}: Caching particle data")
	downs=sorted(set([s[1] for s in stages]))

	# critical for later in the program, this initializes the radius images for all of the samplings we will use
	for d in downs:
		rad_img_int(d)
		if options.ctf>0:
			rad2_img(d)

	### Note that StackCache stores the entire .lst file. selimg must be handled when reading from the cache
	cache=StackCache(args[0])

	if options.verbose>1: print(f"\n{local_datetime()}: Refining")

	# Initialize Points
	point=Points()
	#Initialize Points to random values with amplitudes over a narrow range
	rng = np.random.default_rng()
	rnd=rng.uniform(0.0,1.0,(options.initpoint*9//10,4))		# start with completely random Point parameters
	neg = rng.uniform(0.0, 1.0, (options.initpoint//10, 4))		# 10% of points are negative WRT the background (zero)
	rnd+=(-.5,-.5,-.5,2.0)
	neg+=(-.5,-.5,-.5,-3.0)
	rnd = np.concatenate((rnd, neg))
	point._data=rnd/(1.5,1.5,1.5,3.0)	# amplitudes set to ~1.0, positions random within 2/3 box size

	frchist=[]
	times.append(time.time())
	ptcls=[]
	qualities=np.zeros((nptcl,len(stages)))			# only used with --score
	weights=[None for i in range(len(stages))]		# saved, but not used at present
	thresholds=[None for i in range(len(stages))]	# saved, but not used at present
	for sn,stage in enumerate(stages):
		if options.verbose: print(f"Stage {sn} - {local_datetime()}:")
		if options.profile and sn==2 : jax.profiler.start_trace("jax_trace")

		if options.verbose: print(f"\tIterating x{stage[2]} with frc weight {stage[3]}\n    FRC\t\tshift_grad\tamp_grad\timshift\tgrad_scale")
		lqual=-1.0
		rstep=1.0
		optim = optax.adam(.003)		# parm is learning rate
		optim_state=optim.init(point._data)		# initialize with data
		for i in range(stage[2]):		# training epochs
			if rstep<.01: break		# don't continue if we've optimized well at this level
			if nptcl>stage[0]*2: idx0=sn+i
			else: idx0=0
			nliststg=selimg[range(idx0,nptcl,max(1,nptcl//stage[0]+1))]		# all of the particles to use in the current epoch in the current stage, sn+i provides stochasticity
			imshift=0.0
			step=0.0
			qual=0.0
			shift=0.0
			sca=0.0
			weight=np.ones(stage[1]//2+1)	# defaults in case the weighting estimate below is skipped
			thresh=0.0
			for j in range(0,len(nliststg)-10,batchsize):	# compute the gradient step piecewise due to memory limitations, batchsize particles at a time. The "-10" prevents very small residual batches from being computed
				ptclsfds=cache.read(stage[1],nliststg[j:j+batchsize])	# metadata stored in ptclsfds, which is a EMStack2D
				# Since we aren't modifying the metadata in this program, just make a JAX copy at the get-go
				meta=jnp.array(ptclsfds.metadata)		# 0:ty,1:tx,2:ortx,3:orty,4:ortz,5:defocus,6:phase,7:dfdiff,8:astigangle,9:score,10:class

				if len(ptclsfds)<5 :
					print("Abort tiny batch: ",len(nliststg),j,batchsize)
					continue

				# on the first epoch of each stage we look at the variance of the fsc curve to estimate weighting
				# only using a single batch for this right now. May be sufficient
				if i in (0,8) and j==0:
					frcs=prj_frcs(point.jax,ptclsfds,meta)
					try:
						thresh=1.25*np.std(frcs,0)/math.sqrt(batchsize)
						weight=1.0/np.array(thresh)		# this should make all of the standard deviations the same
						weight[0:2]=0			# low frequency cutoff
						weight[ptclsfds.shape[1]//2:]=0
						weight/=np.sum(weight)	# normalize to 1
						weight=jnp.array(weight*len(weight))	# the *len(weight) is dumb, but due to mean() being returned
					except:
						print(f"Weighting failed {sn},{i},{j}")
						weight=np.ones((ptclsfds.shape[1]//2+1))
					thresholds[sn]=thresh
					weights[sn]=weight
					frchist.append((np.array(np.mean(frcs,0)),thresh,weight))


				if options.ctf==0:
					step0,qual0,shift0,sca0=point_gradient_step_optax(point,ptclsfds,meta,symmx,weight,thresh)
					# TODO: These nan_to_num shouldn't be necessary. Not sure what is causing nans. Could be there is an implicit sqrt somewhere we're taking the gradient of, or a roundoff error?
					step0=jnp.nan_to_num(step0)
					qual0=jnp.nan_to_num(qual0)
					shift0=jnp.nan_to_num(shift0)
					sca0=jnp.nan_to_num(sca0)
					if j==0:
						step,qual,shift,sca=step0,-qual0,shift0,sca0
					else:
						step+=step0
						qual-=qual0
						shift+=shift0
						sca+=sca0
				elif options.ctf==1:
					dsapix=ptclsfds.apix
					wavelength=12.2639/np.sqrt(ptclsfds.voltage*1000.0+0.97845*ptclsfds.voltage*ptclsfds.voltage)
					step0,qual0,shift0,sca0=point_gradient_step_ctf_optax(point,ptclsfds,meta,jnp.array([wavelength, ptclsfds.cs]),dsapix,symmx,weight,thresh)
					step0=jnp.nan_to_num(step0)
					if j==0:
						step,qual,shift,sca=step0,-qual0,shift0,sca0
					else:
						step+=step0
						qual-=qual0
						shift+=shift0
						sca+=sca0
				elif options.ctf==2:
					dsapix=ptclsfds.apix
					wavelength=12.2639/np.sqrt(ptclsfds.voltage*1000.0+0.97845*ptclsfds.voltage*ptclsfds.voltage)
					dfstep=2*apix*apix/(wavelength*10000)
					step0,qual0,shift0,sca0=point_gradient_step_layered_ctf_optax(point,ptclsfds,meta,jnp.array([wavelength, ptclsfds.cs]),dfstep,dsapix,symmx,weight,thresh)
					step0=jnp.nan_to_num(step0)
					if j==0:
						step,qual,shift,sca=step0,-qual0,shift0,sca0
					else:
						step+=step0
						qual-=qual0
						shift+=shift0
						sca+=sca0

			norm=len(nliststg)//batchsize+1
			if norm==0: raise Exception("ERROR: norm zero. This shouldn't happen")
			qual/=norm
			shift/=norm
			sca/=norm
			imshift/=norm

			if math.isnan(float(shift)) or math.isnan(float(sca)):
				if i==0:
					print("ERROR: nan on gradient descent, saving crash images and exiting")
					ims=jnp.fft.irfft2(ptclsfds.jax,s=(ptclsfds.shape[1],ptclsfds.shape[2]))
					io=ImageIO("crash_lastb_images.hdf","rw")
					h2={"nx":ptclsfds.shape[1],"ny":ptclsfds.shape[1],"nz":1,"apix_x":ptclsfds.apix,"apix_y":ptclsfds.apix}
					for k in range(ims.shape[0]):
						io.write_image(-1,np.array(ims[k],dtype=np.float32),h2)
					io.close()
					meta=np.array(meta)
					out=open("crash_lastb_ortdydx.txt","w")
					for k in range(len(meta)):
						out.write(f"{meta[k,2]:1.6f}\t{meta[k,3]*1000:1.6f}\t{meta[k,4]*1000:1.6f}\t{meta[k,0]*1000:1.2f}\t{meta[k,1]*1000:1.2f} (/1000)\n")
					out=None
					sys.exit(1)
				else:
					print("ERROR: encountered nan on gradient descent, skipping epoch. Image numbers saved to crash_img_S_E.lst")
					try: os.unlink("crash_img.lst")
					except: pass
					out=LSXFile(f"crash_img_{sn}_{i}.lst")
					for ii in nliststg: out.write(-1,ii,args[0])
					out.close()
					continue

			update, optim_state = optim.update(step, optim_state, point._data)
			point._data = optax.apply_updates(point._data, update)

			if options.savesteps:
				pd=np.array(point.numpy)
				io=ImageIO("steps.hdf","rw")
				io.write_image(-1,pd,{"nx":4,"ny":pd.shape[0],"nz":1})
				io.close()

			if dbugvol is not None:
				nyd=dbugvol.shape[0]
				if options.sym not in ("c1","C1","I","i"):
					vol=EMStack3D(apply_symmetry(point.volume(nyd,zmax).numpy[0],sym)).do_fft().jax
				else: vol=point.volume(nyd,zmax).do_fft().jax
				fsc=jax_fsc_jit(vol,dbugvol)
				out=open(f"fscm3d_{sn:02d}_{i:02d}.txt","w")
				for s in range(nyd//2): out.write(f"{s/nyd:1.4f}\t{float(fsc[0][s]):1.5f}\n")
				out=None
				dbfsc=float(fsc[0][:nyd//2].mean())
				print(f"{dbfsc:0.5f}\t",end="")

			print(f"{i}\t{qual:1.5f}\t{shift*1000:1.6f}\t\t{sca*1000:1.6f}\t{imshift*1000:1.6f}  # /1000")

			if qual>0.99: break

			# Combine final few iterations to give the final volume
			if options.combineiters>0 and sn == len(stages)-1 and stage[2] - i -1 <= (options.combineiters-1)*refineiters:
				if (i+1-(stage[2]-options.combineiters*refineiters))%refineiters == 0:
					if len(final_point) == 0: final_point._data = np.array(point._data)
					else: final_point._data = np.concatenate([final_point.numpy, np.array(point.jax)], axis=0)
					rng=np.random.default_rng()
					point.coerce_jax()
					std= 1/(2*outsz)
					point._data+=np.hstack((rng.normal(0,std,(point._data.shape[0], 3)), np.zeros((point._data.shape[0], 1))))
					if options.verbose>5 and dbugvol is not None:
						vol=final_point.volume_np(outsz,zmax).center_clip(outsz).numpy[0]
						fscd=compute_fsc(vol,refdata)
						out=open(f"testing_combineiters_fsc_{i}.txt","w")
						for s in range(len(fscd[0])): out.write(f"{s/vol.shape[0]:1.4f}\t{float(fscd[0][s]):1.5f}\n")
						out=None

			# Particle quality assessment
			if options.score is not None and i==stage[2]-1:
				if options.verbose: print("Compute quality")
				for j in range(0,nptcl,batchsize):
					ptclsfds=cache.read(stage[1],nliststg[j:j+batchsize])	# metadata stored in ptclsfds, which is a EMStack2D
					meta=jnp.array(ptclsfds.metadata)		# 0:ty,1:tx,2:ortx,3:orty,4:ortz,5:defocus,6:phase,7:dfdiff,8:astigangle,9:score,10:class

					# need to recompute this here since we may not have hit this stage yet
					if j==0:
						frcs=prj_frcs(point.jax,ptclsfds,meta)
						try:
							thresh=1.25*np.std(frcs,0)/math.sqrt(batchsize)
							weight=1.0/np.array(thresh)		# this should make all of the standard deviations the same
							weight[0:2]=0			# low frequency cutoff
							weight[ptclsfds.shape[1]//2:]=0
							weight/=np.sum(weight)	# normalize to 1
							weight=jnp.array(weight*len(weight))	# the *len(weight) is dumb, but due to mean() being returned
						except:
							print(f"Weighting failed {sn},{i},{j}")
							weight=np.ones((ptclsfds.shape[1]//2+1))

					# we measure per particle quality and save it
					quality=jnp.nan_to_num(sym_prj_frcs_loss(point.jax,symmx,ptclsfds.jax,meta,weight,thresh), nan=2.0)

					for ii,q in enumerate(np.array(quality)):
						if ii+j<nptcl: qualities[ii+j][sn]=q

		# end of epoch, save images and projections for comparison
		if options.verbose>3:
			dsapix=ptclsfds.apix
			pointary=point.jax
			ny=ptclsfds.shape[1]
			orts,tytx=ptclsfds.orientations
			projs=EMStack2D(point_project_simple_sym_fn(pointary, orts.jax, ny, tytx, symmx))
			if options.ctf>0:
				dsapix=ptclsfds.apix
				ctf=ptclsfds.ctf
				wavelength=12.2639/np.sqrt(ptclsfds.voltage*1000.0+0.97845*ptclsfds.voltage*ptclsfds.voltage)
				dfstep=2*apix*apix/(wavelength*10000)
				ctf_projs=EMStack2D(point_project_ctf_sym_fn(pointary, orts.jax, jnp.array([wavelength,ptclsfds.cs]), dsapix, ny, tytx, ctf, 0.0, symmx))
				layered_ctf_projs=EMStack2D(point_project_layered_ctf_sym_fn(pointary,orts.jax,jnp.array([wavelength,ptclsfds.cs]),dfstep,dsapix,ny,tytx,ctf, symmx))
			ptclds=EMStack2D(jnp.fft.irfft2(ptclsfds.jax,s=(ny,ny)))
			transforms=orts.transforms(tytx=tytx)
			io=ImageIO(f"debug_img_{ny}.hdf","rw")
			hd={"nx":ny,"ny":ny,"nz":1,"apix_x":dsapix,"apix_y":dsapix}
			for k in range(len(projs)):
				a=ptclds.numpy[k].copy()
				b=projs.numpy[k].copy()
				h=dict(hd); h["xform.projection"]=transforms[k].to_jsondict()
				a=a-a.mean()
				if a.std()<1e-12: a=a+1e-12
				else: a=a/a.std()
				def matchto(im):
					im=im-im.mean()
					s=im.std()
					if s<1e-12: return im
					return im*(a.std()+1e-12)/s
				b=matchto(b)
				if options.ctf>0:
					c=matchto(ctf_projs.numpy[k])
					d=matchto(layered_ctf_projs.numpy[k])
				io.write_image(-1,a.astype(np.float32),h)
				io.write_image(-1,b.astype(np.float32),h)
				if options.ctf>0:
					io.write_image(-1,c.astype(np.float32),h)
					io.write_image(-1,d.astype(np.float32),h)
			io.close()

		# filter results and prepare for stage 2
		if stage[5]>0:			# no filter/replicate in the last stage
			g0=len(point)
			point.norm_filter(sig=stage[4],rad_downweight=0.33)
			g1=len(point)
			# Replicate points to produce a specified total number for each stage. Critical that these numbers
			# fall in a small set of specific N for JIT compilation
			point.replicate_abs(stage[5]*options.initpoint,stage[6])
			g2=len(point)
		else: g0=g1=g2=len(point)
		print(f"{local_datetime()}: Stage {sn} complete: {g0} -> {g1} -> {g2} points")
		times.append(time.time())

		# do this at the end of each stage in case of early termination
		if options.pointout is not None and g2 != 0:
			np.savetxt(options.pointout,point.numpy,fmt="%0.4f",delimiter="\t")

	if options.profile : jax.profiler.stop_trace()

	times.append(time.time())

	if options.score is not None:
		np.savetxt(options.score.replace(".lst","_score.txt"),qualities,fmt="%.6f")
		np.savetxt("scores_t.txt",qualities.transpose(),fmt="%.6f")

		lsxout=LSXFile(options.score)
		for i in range(nptcl):
			n,f,d=lsx[selimg[i]]
			d["score"]=float(qualities[i][-1])
			lsxout[i]=n,f,d
		lsxout.close()

	if options.combineiters>0:
		if options.verbose>5: np.savetxt("testing_combine_iters.txt", final_point.numpy, fmt="%0.4f", delimiter="\t") # For testing
		if options.postclip>0: vol = final_point.volume_np(outsz,zmax).center_clip(options.postclip)
		else: vol=final_point.volume_np(outsz,zmax).center_clip(outsz)
	elif options.postclip>0 : vol=point.volume(outsz,zmax).center_clip(options.postclip)
	else : vol=point.volume(outsz,zmax).center_clip(outsz)
	vol=vol.numpy[0]
	if options.sym not in ("c1","C1","I","i"):
		if options.verbose>0 : print(f"Apply {options.sym} symmetry to map (not points)")
		vol=apply_symmetry(vol,sym)
	times.append(time.time())
	nhdr={"apix_x":apix*nxraw/outsz,"apix_y":apix*nxraw/outsz,"apix_z":apix*nxraw/outsz,"sym":options.sym}
	if options.ptcl3d_id is not None : nhdr["ptcl3d_id"]=options.ptcl3d_id
	write_volume(options.volout.replace(".hdf","_unfilt.hdf"),vol,nhdr)
	if options.volfiltlp>0: vol=lowpass_gauss(vol,1.0/options.volfiltlp)
	vol=normalize_edgemean(vol)
	times.append(time.time())
	write_volume(options.volout,vol,nhdr)

	outf=open("frcstats.txt","w")
	for i in range(len(frchist[-1][0])):
		outf.write(f"{i}")
		for j in range(len(frchist)):
			try: outf.write(f"\t{frchist[j][0][i]:1.6f}\t{frchist[j][1][i]:1.6f}\t{frchist[j][2][i]:1.6f}")
			except:
				outf.write("\t0.0\t0.0\t0.0")
		outf.write("\n")
	outf.close()

	# FSC against the reference volume (replaces the old e2proc3d.py --calcfsc call)
	if options.fscdebug is not None:
		fsc=compute_fsc(np.asarray(vol),refdata)
		outf=open(f"{options.volout.rsplit('.',1)[0]}_fsc.txt","w")
		for s in range(len(fsc[0])):
			outf.write(f"{s/vol.shape[0]:1.4f}\t{float(fsc[0][s]):1.5f}\n")
		outf.close()

	times=np.array(times)
	times=times[1:]-times[:-1]
	if options.verbose>1 : print(times.astype(np.int32))

	E3end(llo)


if __name__=="__main__":
	main()
