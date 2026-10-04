# EMAN3
EMAN3 is the next major step in the evolution of EMAN. It is now a functional standalone package. It does not replicate all of the functionality of EMAN2, but focuses on the specific developmental paths we've followed in recent years. We are still developing some of the key tools, so there are no tutorials for any of the packages yet, but we are finally making rapid progress on this.

EMAN3 is a pure python package. The only C++ code comes from separately distributed dependencies (like jax, numpy, etc.). This means it can be trivially installed in just a few minutes.

At this moment, we recommend installing EMAN3 in a separate conda or venv environment from EMAN2. While they are generally compatible and share many dependencies, EMAN3 uses JAX for its deep learning tools, and much of EMAN2 is still based on TensorFlow. It can sometimes be challenging to get Tensorflow and JAX working in the same environment at the same time.

We have not created a PYPI (pip) package for EMAN3 yet, as it is still pre-alpha, but installation is pretty trivial:

To install:
```
# create environment and install dependencies
conda create -n eman3 -c conda-forge pyside6 scipy matplotlib pygfx h5py
conda activate eman3
# Linux:
pip install -U "jax[cuda13]" optax orbax-checkpoint tifffile flax
# Mac:
pip install -U jax optax orbax-checkpoint tifffile flax

# Get EMAN3 and install
git clone https://github.com/cryoem/eman3.git
cd eman3
pip install -e .
```

EMAN3:
* programs begin with "e3" instead of "e2", and executables do not require the .py extension
* is written in Python on top of NumPy and JAX. Almost everything is GPU accelerated (wherever it makes sense)
* uses pygfx with pyside6 for GUI tools (instead of OpenGL with PyQt5)
* EMAN3 interacts with Relion and CryoSparc much more easily. Simple conversion for comparative work.
* Image data stored in EMAN3 is almost always compressed [PMC9645247](https://pmc.ncbi.nlm.nih.gov/articles/PMC9645247/), requiring less storage and improving speed without compromising quality.
* As with EMAN2, using --help with any program should give a summary of usage and command line options

Current usable/useful programs include:
* e3display - very similar to e2display, but should be more responsive with the new GUI system and caching system
* e3make3d_point - performs a point-based 3-D reconstruction

Other programs are not yet in a useful state, or are still part of the EMAN2 distribution.

6/5/25 - At present, EMAN3 is installed automatically with EMAN2. When this changes, this comment will be updated.
10/3/26 - EMAN3 is now completely standalone, if incomplete
