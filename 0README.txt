To install immediately from local source: pip install -e .
For a hard reinstall: pip install --force-reinstall --no-deps -e .

to build a wheel: python -m build .
pip install dist/eman3-0.1.0-py3-none-any.whl

to create a conda environment:
conda create -n eman3 -c conda-forge pyside6 scipy matplotlib pygfx h5py
conda activate eman3
pip install -U "jax[cuda13]" optax orbax-checkpoint tifffile flax
