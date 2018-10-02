"""

ToDo: NPT Simulations in LAMMPS

BUG: LammpsEngine._put_boxvector(self, nparray) -> Inf Pressure
"""
import os
import numpy as np
import simtk.unit as u

import openpathsampling as paths
from openpathsampling.engines import DynamicsEngine
from .snapshot import Snapshot

from openpathsampling.engines.openmm.topology import MDTrajTopology

from lammps import lammps

import sys

if sys.version_info > (3,):
    basestring = str


class LammpsEngine(DynamicsEngine):
    """OpenMM dynamics engine based on a openmmtools.testsystem object.

    This is only to allow to use the examples from openmmtools.testsystems
    within this framework
    """

    # units = {
    #     'length': u.dimensionless,
    #     'velocity': u.dimensionless,
    #     'energy': u.dimensionless
    # }

    _default_options = {
        'n_steps_per_frame': 10,
        'n_frames_max': 5000
    }

    def __init__(self, inputs, options=None, topology=None, lmp_name=None, lmp_cmdargs=None):
        """

        Parameters
        ----------
        inputs
        options
        topology
        lmp_name : str or None
            Name of the LAMMPS library (default is None)

            `lmp_name='gpu'` create a LAMMPS object using the `liblammps_gpu.so` library.
        lmp_cmdargs : List(str) or None
            command-line arguments passed to LAMMPS, e.g. list = ["-echo","screen"]
            or for gpu ['-sf', 'gpu', '-pk', 'gpu', '1']

            see: https://lammps.sandia.gov/doc/Run_options.html
        """

        self.inputs = inputs

        self.mdtraj_topology = None
        if topology is not None:
            topology = self._get_topology(topology)
            self.mdtraj_topology = topology.mdtraj
        self.topology = topology

        # Create new lammps instance
        self._lmp = lammps(name=lmp_name, cmdargs=lmp_cmdargs)

        # Execute the give script
        commands = inputs.splitlines()

        for command in commands:
            self._lmp.command(command)

        # self.command('compute thermo_ke all ke')
        self.command('run 0')

        template = self._get_snapshot(True)

        dimensions = {
            'n_atoms': template.coordinates.shape[0],
            'n_spatial': template.coordinates.shape[1]
        }

        descriptor = paths.engines.SnapshotDescriptor.construct(
            Snapshot,
            dimensions
        )

        super(LammpsEngine, self).__init__(
            options=options,
            descriptor=descriptor

        )

        # set no cached snapshot, means it will be constructed
        # from the openmm context
        self._current_snapshot = None
        self._current_kinetics = None
        self._current_statics = None
        self._current_box_vectors = None
        self._box_center = None
        # TODO: so far we will always have an initialized system which we should change somehow
        self.initialized = True

    @staticmethod
    def _get_topology(topology):
        # topology can be a filename or an MDTrajTopology
        try:
            import mdtraj as md
        except ImportError:
            raise RuntimeWarning("Missing MDTraj; topology keyword ignored")
        else:
            if isinstance(topology, MDTrajTopology):
                return topology

            if isinstance(topology, basestring) and os.path.isfile(topology):
                topology = md.load(topology).topology

            if isinstance(topology, md.Topology):
                return MDTrajTopology(topology)
            else:
                return None  # may later allow other ways

    def command(self, *args, **kwargs):
        self._lmp.command(*args, **kwargs)

    def create(self):
        """
        Create the final OpenMMEngine

        """
        self.initialized = True

    def _get_snapshot(self, topology=None):
        lmp = self._lmp
        x = lmp.gather_atoms("x", 1, 3)
        images = lmp.gather_atoms("image", 0, 3)
        v = lmp.gather_atoms("v", 1, 3)

        (xlo, ylo, zlo), \
            (xhi, yhi, zhi), \
            xy, yz, xz, \
            periodicity, box_change = lmp.extract_box()

        bv = np.array(
            [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]])
        self._box_center = bv[(0, 1, 2), (0, 1, 2)] + np.array([xlo, ylo, zlo])
        n_atoms = lmp.get_natoms()
        # n_spatial = len(x) / n_atoms
        x = np.array(np.ctypeslib.array(x).reshape((n_atoms, -1)))
        v = np.ctypeslib.array(v).reshape((n_atoms, -1))
        images = np.ctypeslib.array(images).reshape((n_atoms, -1))
        x += np.dot(images, bv)
        snapshot = Snapshot.construct(
            engine=self,
            coordinates=x * u.angstrom,
            box_vectors=bv * u.angstrom,
            velocities=v * u.angstrom / u.femtosecond
        )

        return snapshot

    def _put_coordinates(self, nparray):
        nparray = np.asarray(nparray.in_units_of(u.angstrom), dtype=np.float64)
        lmp = self._lmp
        # reset images to zeros
        # ToDo: test performance set vs scatter
        lmp.command('set group all image 0 0 0')
        # images_zero = np.zeros((lmp.get_natoms(), 3), dtype=np.int32)
        # lmparray_images = np.ctypeslib.as_ctypes(images_zero.ravel())
        # lmp.scatter_atoms("image", 0, 3, lmparray_images)
        lmparray = np.ctypeslib.as_ctypes(nparray.ravel())
        lmp.scatter_atoms("x", 1, 3, lmparray)

    def _put_velocities(self, nparray):
        nparray = np.asarray(nparray.in_units_of(u.angstrom / u.femtosecond), dtype=np.float64)
        lmp = self._lmp
        lmparray = np.ctypeslib.as_ctypes(nparray.ravel())
        lmp.scatter_atoms("v", 1, 3, lmparray)

    def _put_boxvector(self, nparray):
        """
        Sets the box vector.
        Needed for npt simulations.

        Parameters
        ----------
        nparray : np.array
            box_vectors
        """
        nparray = np.asarray(nparray.in_units_of(u.angstrom), dtype=np.float64)
        boxdim = nparray[np.arange(nparray.shape[0]),np.arange(nparray.shape[0])]
        if self._box_center is None:
            self._box_center = np.zeros(3)
        boxlo = - np.asarray(boxdim)  / 2.0 + self._box_center
        boxhi = - np.asarray(boxdim) / 2.0 + self._box_center
        xy = float(nparray[1,0])
        xz = float(nparray[2,1])
        yz = float(nparray[2,1])
        self._lmp.reset_box(boxlo,boxhi,xy,yz,xz)

    @property
    def lammps(self):
        return self._lmp

    def to_dict(self):
        return {
            'inputs': self.inputs,
            'options': self.options,
            'topology': self.topology
        }

    @property
    def snapshot_timestep(self):
        if 'timestep' not in self.options:
            return self.n_steps_per_frame * self._lmp.get_thermo('dt')
        return self.n_steps_per_frame * self.options['timestep']

    def _build_current_snapshot(self):
        return self._get_snapshot()

    @property
    def current_snapshot(self):
        if self._current_snapshot is None:
            self._current_snapshot = self._build_current_snapshot()

        return self._current_snapshot

    def _changed(self):
        self._current_snapshot = None

    @current_snapshot.setter
    def current_snapshot(self, snapshot):
        if snapshot is not self._current_snapshot:
            if snapshot.statics is not None:
                if self._current_snapshot is None or snapshot.statics is not self._current_snapshot.statics:
                    # new snapshot has a different statics so update
                    self._put_coordinates(snapshot.coordinates)

            if snapshot.kinetics is not None:
                if self._current_snapshot is None or snapshot.kinetics is not self._current_snapshot.kinetics or snapshot.is_reversed != self._current_snapshot.is_reversed:
                    self._put_velocities(snapshot.velocities)

            #if snapshot.box_vectors is not None:
                # update box
                #self._put_boxvector(snapshot.box_vectors)

            # After the updates cache the new snapshot
            self._current_snapshot = snapshot
            # reinitialize LAMMPS (neighbour list etc) with the new coordinates
            self._lmp.command('run 0')

    def run(self, steps):
        self._lmp.command('run ' + str(steps))

    def generate_next_frame(self):
        # disable pre & post processing for continues runs
        self._lmp.command('run ' + str(self.n_steps_per_frame) + ' pre no post no')
        self._current_snapshot = None
        return self.current_snapshot

    @property
    def kinetics(self):
        return self.current_snapshot.kinetics

    @property
    def statics(self):
        return self.current_snapshot.statics

    def minimize(self, etol=1e-6, ftol=1e-6, maxiter=100, maxeval=1000):
        # type: (LammpsEngine, float, float, int, int) -> None
        """
        Energy minimization using a conjugated gradient.

        LAMMPS-Syntax
        -------------
        `min_style cg`

        `min_modify line quadratic`

        `minimize {etol} {ftol} {maxiter} {maxiter}`

        `reset_timestep 0`


        Parameters
        ----------
        etol : float
            stopping tolerance for force (force units)
        ftol : float
            stopping tolerance for force (force units)
        maxiter : int
            max iterations of minimizer
        maxeval : int
            max number of force/energy evaluations

        """
        self._lmp.commands_list(["min_style cg",
                                 "min_modify line quadratic",
                                 "minimize {etol} {ftol} {maxiter} {maxeval}".format(
                                     etol=etol, ftol=ftol,
                                     maxiter=maxiter, maxeval=maxeval),
                                 "reset_timestep 0"])

        # make sure that we get the minimized structure on request
        self._current_snapshot = None
