# Installation

Directly from github using pip

    $ pip install git+ssh://git@github.com/Subaru-PFS-GA/ga_isochrones.git

No package in PYPI

# Creating packages

## Creating a pip package

In project root, run

    $ python setup.py build

To clean up everything, run

    $ python setup.py clean

To install as a module using pip, run

    $ pip install .

To uninstall, run

    $ pip uninstall pfs-isochrones