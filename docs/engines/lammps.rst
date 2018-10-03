.. _lammps:

.. module:: openpathsampling.engines.lammps

LAMMPS OPS Engine
=================

Primary Classes
---------------

.. autosummary::
   :toctree: ../api/generated/

   Snapshot
   MDSnapshot
   Engine


Utility functions
-----------------

.. autosummary::
   :toctree: ../api/generated/

   snapshot_from_pdb
   snapshot_from_testsystem
   trajectory_from_mdtraj
   empty_snapshot_from_openmm_topology
   to_openmm_topology



LAMMPS as library
------------------------------
Checkout: https://lammps.sandia.gov/doc/Python_library.html


How To Build LAMMPS as library
------------------------------

This is an example how to build the ``liblammps.so`` file needed for the LAMMPS python interface.

A full description how to build LAMMPS can be found under:
* https://lammps.sandia.gov/doc/Build.html
* ``cmake``
   * https://lammps.sandia.gov/doc/Build_cmake.html
   * https://github.com/lammps/lammps/blob/master/cmake/README.md

cmake
^^^^^
activate anaconda::

   ANACONDA_PATH=${HOME}/anaconda3
   source ${ANACONDA_PATH}/bin/activate

set ``cmake`` flags::

   # -D LAMMPS_MACHINE=plumed \
   CMAKE_FLAGS=" \
    -D CMAKE_INSTALL_PREFIX=${ANACONDA_PATH} \
    -D BUILD_OMP=yes \
    -D WITH_JPEG=yes \
    -D WITH_PNG=yes \
    -D WITH_GZIP=yes \
    -D LAMMPS_EXCEPTIONS=no \
    -D PKG_COMPRESS=yes \
    -D PKG_MPIIO=yes \
    -D PKG_PYTHON=yes \
    -D PKG_USER-MISC=yes \
    -D PKG_USER-MOLFILE=yes \
    -D PKG_USER-OMP=yes \
    -D PKG_GPU=yes \
    -D GPU_API=cuda \
    -D GPU_ARCH=sm_61 \
    -D LAMMPS_EXCEPTIONS=yes \
    -D BUILD_LIB=yes \
    -D BUILD_SHARED_LIBS=yes \
    -D BUILD_EXE=no \
   "

create a build directory and build ``liblammps.so``::

   mkdir build; cd build
   cmake -C  ../cmake/presets/std_nolib.cmake  ${CMAKE_FLAGS} ../cmake
   make -j 6
   make install

create a lammps executable for normal simulations::

   cmake -C ../cmake/presets/std_nolib.cmake ${CMAKE_FLAGS} ../cmake
   cmake -D LAMMPS_EXCEPTIONS=no -D BUILD_LIB=no -D BUILD_SHARED_LIBS=no -D BUILD_EXE=yes ../cmake
   make -j 6
   make install

