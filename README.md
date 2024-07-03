# testing-artefacts-pharmaco-multipath

# About
Testing artefacts in pharmaceutically-controlled multi-pathogen systems: a modelling study. The project is mainly run by Priesemann Group at MPIDS, under the supervision of Viola Priesemann.
Main contributors: 

# File Structure

## Directories
- testing-artefacts-pharmaco-multipath: all the scripts are inside this directory. Note: adding new scripts? Don't forget to update the __init__.py
- results: 
- notebooks: all jupyter notebooks are here

# Installation

## From sources
The sources for icomo can be downloaded from
the [Github repo](https://github.com/Priesemann-Group/icomo.git)


Clone the public repository:

```bash
$ git clone https://github.com/Priesemann-Group/testing-artefacts-pharmaco-multipath.git
```
or (better, requires ssh key setup)
```bash
$ git clone git@github.com:Priesemann-Group/testing-artefacts-pharmaco-multipath.git
```

Enter the directory and install with pip:

```bash
$ cd testing-artefacts-pharmaco-multipath
$ pip install -e .
```
This enables you to edit the code and have the changes directly available in your python
environment.

You might also want to install [jupyterlab](https://jupyter.org) to run the notebooks:

```bash
$ pip install jupyterlab
```

# Development notes

## Add dependencies

Dependencies are listed in pyproject.toml. To add a new dependency, add it to the list in pyproject.toml 
and run
```bash
$ pip install -e .
```
to install the new dependencies in your environment.
