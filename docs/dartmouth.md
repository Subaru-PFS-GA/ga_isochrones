# Download and uncompress data

Download isochrone tables from http://stellar.dartmouth.edu/models/
Isochrones with synthetic photometry come in tgz archives, once downloaded, uncompress them, e.g.

    $ mkdir mist && cd mist
    $ wget http://stellar.dartmouth.edu/models/isochrones/PanSTARRS.tgz
    $ tar fxvz PanSTARRS.tgz

# Import into HDF5 format

To speed up data loading, the original ASCII isochrones tables should be converted into a binary HDF5 file.

    $ pfsiso-import dartmouth --in <input_dir> --out <output_dir> --photometry <photometry1> <photometry2> --alpha <alpha> --helium <vvcrit>

where

* `<input_dir>` is the directory where the isochrone files have been decompressed to. (This is a directory containing further directories with names of the photometric systems)
* `<output_dir>` is the path to the directory where the resulting files (HDF5 and logs) will be created.
* `<photometry>` is the filename postfix of the photometric system, such as SDSSugriz, see the Dartmouth web page.
* `<alpha>` is the value of [a/Fe] using the Dartmouth notation, i.e. p0 for Solar.
* `<helium>` is the value of Y, omit or use 33 or 44.

As a concrete example, try the command line

    $ pfsiso-import dartmouth --in ./isochrones/dartmouth --out ./temp/dartmouth_sdss --photometry SDSSugriz --alpha p0

providing that the isochrone files are extracted under `./isochrones/dartmouth`.

Note that the output set by `--out` is not a file but a directory containing several files. The output directory must not exist in order to avoid silent overwriting of important data.

In addition to importing Dartmouth isochrones in the precomputed photometric systems, Hyper Supreme-Cam photometry can be computed on-the-fly by specifying `HSC` as an option, after `SDSSugriz`, to `--photometry`:

    $ pfsiso-import dartmouth --in ./isochrones/dartmouth --out ./temp/dartmouth_sdss_cfht_hsc --photometry SDSSugriz CFHTugriz HSC --alpha p0