# Download and uncompress data

Download isochrone tables from http://waps.cfa.harvard.edu/MIST/model_grids.html
Isochrones with synthetic photometry come in txz archives, once downloaded, uncompress them, e.g.

    $ mkdir mist && cd mist
    $ wget http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_SDSSugriz.txz
    $ tar fxvJ MIST_v1.2_vvcrit0.4_SDSSugriz.txz

# Import into HDF5 format

To speed up data loading, the original ASCII isochrones tables should be converted into a binary HDF5 file.

    $ pfsiso-import mist --in <input_dir> --out <output_dir> --photometry <photometry> --a_Fe <alpha> --v_v_crit <vvcrit>

where

* `<input_dir>` is the directory where the isochrone files have been decompressed to. (This is a directory containing further directories with names like MIST_v1.2_vvcrit0.4_SDSSugriz!)
* `<output_dir>` is the path to the directory where the resulting files (HDF5 and logs) will be created.
* `<photometry>` is the filename postfix of the photometric system, such as SDSSugriz, see the MIST web page.
* `<alpha>` is the value of [a/Fe] using the MIST notation, i.e. p0.0 for Solar.
* `<vvcrit>` is the value of v/v_crit, 0.0 or 0.4

As a concrete example, try the command line

    $ import mist --in ./isochrones/mist --out ./temp/mist_sdss --photometry SDSSugriz  --a_Fe p0.0 --v_v_crit 0.4

providing that the isochrone files are extracted under `./isochrones/mist`.

Note that the output set by `--out` is not a file but a directory containing several files. The output directory must not exist in order to avoid silent overwriting of important data.