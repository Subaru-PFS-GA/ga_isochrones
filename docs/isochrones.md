# The PFS ISOCHRONES package - Overview

## Introduction

This package is capable of:

* very fast interpolation of isochrone tables on the GPU
* when PFSSPEC is also installed, calculating synthetic magnitudes for theoretical isochrones from synthetic stellar spectrum grids.

## Isochone interpolation

Since the stellar evolutionary phases (hence observable parameters) of two stars with just slightly different initial masses can be very different at evolved stages, interpolation of isochrones in the (age, metallicity, initial mass) domain is not feasible. Instead, interpolation is done as a function of Equivalent Evolutionary Phase (EEP) as opposed to initial mass. In this case, initial mass is just another parameter that is tabulated as a function of the triplet (age, metallicity, initial mass). As a result, when a star is parameterized with the triplet (age, metallicity, initial mass), interpolation in (age, metallicity, EEP) becomes and implicit problem as the algorithm has to find the best EEP which results in the correct initial mass.

The PFS ISOCHRONES library is capable of interpolating any isochrone grid as long as it's given as a function of EEP, such as the grids derived from the MIST theoretical evolutionary tracks and the Darthmouth semi-empirical library.

## Command-line executables

The package installs the following command-line scripts. Read the upcoming sections for details on them.

* **pfsiso-import**: Imports data files such as isochrone tables into HDF5 format that is readily usable by PFS ISOCHRONES.
* **pfsiso-synthmag**: Computes synthetic magnitudes for an already imported isochrone grid from synthetic stellar grid. This executable depends on the PFSSPEC package.

When running from "source" in the development environment (see below), these executables are available from the source root directory as `.\bin\import`, etc. i.e. without the `pfsiso-*` prefix.

## GPU support

The PFS ISOCHRONES package takes advantage of GPU accelleration to significantly speed up isochrone interpolation.

PFS ISOCHRONES uses TensorFlow in eager mode. This means that a wide range of accellerators with Tensorflow support are available including GPUs as well as multi-core CPUs.

## Limitations

* Isochrone interpolation is done for metallicity, age and EEP (or initial mass) only, no support for additional parameters such as $[a/M]$.

## Feedback

Please provide feedback on the package on the GitHub issues page:

* https://github.com/Subaru-PFS-GA/ga_isochrones/issues

# Installation

## Installation as a package

Using `pip` the package can be installed directly from its github repository:

    $ pip install ssh+git@github.com/Subaru-PFS-GA/ga_isochrones.git

Currently no PYPI or Anaconda packages are available.

## Setting up the development environment

TBW

# Isochrone libraries

The PFS ISOCHRONES interpolation package requires isochrone libraries precomputed as a function of the population paramers metallicity and age as well as Equivalent Evolutionary Phase (EEP) indicators to interpolate the magnitudes of evolved star reliably. The currently supported isochrone libraries are MIST and Dartmouth.

## Importing isochrone libraries

The command `isochrones-import` is used to import isochrone libraries from their original data format. 

Dartmouth isochrones can be imported into HDF5 using the command

    $ pfsiso-import dartmouth --in <path_to_data>/isochrones --out <output_dir> --alpha p0 --photometry <photometric_system1> <photometric_system2>

This will look for data files in the input directory in the same directory structure in which the Dartmouth isochrone files are originally provided:

    ./isochrones/<photometric_system>/feh***afep0*

All output files, including 'isochrones.h5' will be placed in the output directory. The argument `--photometry` can list any number of photometric systems as long as they are available under the input directory. Including only the necessary photometric data in isochrone tables, however, will reduce load time and GPU memory usage.

## Synthetic magnitudes for isochrone tables

CMDFIT can compute synthetic magnitudes from theoretical isochrones and spectrum grids. This feature also requires the PFSSPEC modules ˙core` and `stellar`. The following command calculates additional photometry for an existing isochrone table:

    $ pfsiso-synthmag --isogrid-type dartmouth --isogrid-path <isochrone_directory> --specgrid-type phoenix --specgrid-path <specgrid_directory> --filter-names hsc_g2 hsc_i2 hsc_nb515 --filter-files <filter_file1> <filter_file2> ... --out <output_dir>

Any number of filters (filter names and filter files) can be listed here but filter names must be unique and different from the filter names already present in the input isochrone grid.

## Isochrone file data format

The typical structure of an HDF5 isochrones files is the following (actual output from `h5ls`; array sizes may vary):

    /                        Group
    /isochrones              Group
    /isochrones/axes         Group
    /isochrones/axes/EEP     Dataset {369}
    /isochrones/axes/Fe_H    Dataset {9}
    /isochrones/axes/log_t   Dataset {52}
    /isochrones/values       Group
    /isochrones/values/M_ini Dataset {9, 52, 369}
    ...    
    /isochrones/values/log_T_eff Dataset {9, 52, 369}
    /isochrones/values/log_L Dataset {9, 52, 369}
    /isochrones/values/log_g Dataset {9, 52, 369}
    ...
    /isochrones/values/hsc_g Dataset {9, 52, 369}
    /isochrones/values/hsc_i Dataset {9, 52, 369}
    ...
    /isochrones/values/sdss_g Dataset {9, 52, 369}
    /isochrones/values/sdss_i Dataset {9, 52, 369}
    /isochrones/values/sdss_r Dataset {9, 52, 369}
    /isochrones/values/sdss_u Dataset {9, 52, 369}
    /isochrones/values/sdss_z Dataset {9, 52, 369}

All three axes, the `M_ini` data array and at least a single magnitude is necessary for CMDFIT. In this example, magnitudes for two photometric systems, HSC and SDSS are available. The naming of the magnitudes is arbitrary but names should not collide with names of the physical parameters and prefixing them with the name of the photometric system is a good practice. Names are case-sensitive.

All data arrays are float numbers. They're converted to the required 32 or 64 bit precision, when executing the calculations on the GPU. Isochrone libraries often omit early and late evolutionary phases for certain EEPs (or the corresponding initial mass ranges) when no good models are available or the phases are physically irrelevant. These are flagged by storing the special IEEE 754 floating point values -Inf (early stages) or Inf (late stages) in the data array elements corresponding to a combination of (metallicity, age, EEP). CMDFIT uses the values stored in the `M_ini` array to determine if a particular evolutionary phase is available in the library or not.