# PFS ISOCHRONES

This package was meant for astronomers, it accellerates isochrone interpolation on the GPU and capable of interpolating magnitudes and physical parameters of millions of stars in under a second. (Actual number of stars and performance are limited by GPU memory and performance.) Interpolation is based on Equivalent Evolutionary Phase (EEP).

The library currently supports the following isochrone grids:

* MIST: http://waps.cfa.harvard.edu/MIST/
* Dartmouth: http://stellar.dartmouth.edu/models/

The interpolation algorithm is loosely based on

* https://github.com/timothydmorton/isochrones

but adopted for vectorized, GPU-based execution.

See above links for journal references.

# Installation

The package installs its dependencies automatically except for optional dependencies. An important but complicated dependency is TensorFlow and even though it is installed automatically as well, we recommend installing it manually into a new anaconda environment, otherwise GPU acceleration might not work properly.

Create a new conda environment with

    $ conda create -n isochrones python=3.9

Activate the environment with

    $ conda activate isochrones

Install all necessary packages, e.g.

    $ conda install pip numpy scipy tensorflow

Finally, run pip to install the PFS ISOCHRONES package

    $ pip install git+ssh://git@github.com/Subaru-PFS-GA/ga_isochrones.git

You need to have an SSH key set up for GitHub access. Currently, no package is available from PYPI or any Anaconda channels.

# Developer environment

After cloning the repository, create the default environment configuration file  `./configs/envs/default` using `./configs/envs/example` as a basis. After opening a terminal to the project source directory, source the initialization script:

    $ source bin/init

This will configure everything automatically for PFS ISOCHRONES development.

## Unit tests

Python `unittest` tests are available under `./test` but require data files that are not part of the source repository.

## Building packages

From the project root, run

    $ ./bin/setup conda-build

It will assemble the library and put the conda package under `./build/pfs-isochrones`.