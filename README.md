# EMAN3
EMAN3 is the next major step in the evolution of EMAN. At this point it is not a complete standalone package, and must be installed within an existing [EMAN2](https://eman2.org/ "EMAN2") conda environment. 
EMAN3 will gradually evolve towards being a completely standalone package. There are some key differences between EMAN2 and EMAN3:

* EMAN3 programs begin with "e3" instead of "e2"
* EMAN3 is written in Python on top of NumPy and JAX. Almost everything is GPU accelerated (wherever it makes sense)
* EMAN3 interacts with Relion and CryoSparc much more easily. Simple conversion for comparative work.
* Image data stored in EMAN3 is almost always compressed [PMC9645247](https://pmc.ncbi.nlm.nih.gov/articles/PMC9645247/), requiring less storage and improving speed without compromising quality.
* As with EMAN2, using --help with any program should give a summary of usage and command line options
* EMAN3 does not yet have its own GUI, and relies on EMAN2 for visualization

Current usable/useful programs include:
* e3cryosparc_convert.py - Converts a saved CryoSparc project into an EMAN3 project with as much metadata preserved as possible
* e3relion_convert.py - Converts a Relion STAR file and accompaning image data from Relion to EMAN3. Primarily aimed at single particle refinements completed in Relion for comparitive work, some STAR files may be incompatible.
* e3movie.py - _(gpu)_ For processing movies collected on direct detectors, can do something similar to VanHeel-style gain correction, and performs GPU accelerated whole frame movie alignment. Usable, but feature-incomplete.
* e3make3d_gauss.py - _(gpu)_ New Gaussian-based 3-D reconstruction algorithm, which can produce extremely significant improvements over traditional Fourier reconstructions, particularly for particles with preferred orientations. This does not refine particle ortientations, it simply takes an existing set of particles and orientations and performs a 3-D reconstruction.
* e3proc2d.py - _(gpu)_ General purpose image processing and file conversion similar to e2proc2d.py, but with dramatically faster read/write. Usable/useful but feature-incomplete.
* e3extract_subunit.py - _(gpu)_ Based on a preliminary Gaussian model (e3make3d_gauss.py), subtract the model density from each individual particle image except in a defined region for (potentially) improved local refinements.
* e3spa_refine_gauss.py - _(gpu)_ **(experimental)** Combined Gaussian 3-D reconstruction and orientation refinement.

Other programs are not yet in a useful state.

6/5/25 - At present, EMAN3 is installed automatically with EMAN2. When this changes, this comment will be updated.
