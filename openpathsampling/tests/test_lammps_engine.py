"""
@author Michael King
"""
from __future__ import division
from __future__ import absolute_import

from builtins import range
from builtins import object

from nose.tools import (assert_equal)

import numpy as np
import os
from openpathsampling.engines import lammps as ops_lammps

from .test_helpers import (
    true_func, data_filename,
    assert_equal_array_array,
    assert_not_equal_array_array,
    raises_with_message_like)


class TestLAMMPSEngine(object):
    def setup(self):

        self.positions_start = positions_start = np.array(
            [[0, 0, 0], [0, 0, 1]],
            dtype=np.float64)
        self.images_start = np.zeros(positions_start.shape)
        self.pdb_file = 'test_lammps.pdb'

        int_system = [
            'units real',
            'atom_style full',
            'region box block -5.0 5.0 -5.0 5.0 -5.0 5.0',
            'create_box 1 box bond/types 1 extra/bond/per/atom 1',
            'mass 1 1.0',
            'create_atoms 1 single {} {} {}'.format(*self.positions_start[0]),
            'create_atoms 1 single {} {} {}'.format(*self.positions_start[1]),
            'create_bonds single/bond 1 1 2',
            'pair_style zero 5.0 nocoeff',
            'pair_coeff * *',
            'bond_style harmonic',
            'bond_coeff 1 100.0 1.0',
            'run 0'
        ]

        # create a dummy pdb file
        pdb_file = [
            "CRYST1   12.000   12.000   12.000  90.00  90.00  90.00 P 1          1",
            "HETATM    1  C1  XXX A   1    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C".format(
                *self.positions_start[0]),
            "HETATM    2  C2  XXX A   2    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C".format(
                *self.positions_start[1]),
            "END"
        ]
        with open(self.pdb_file, 'w') as fp:
            fp.write('\n'.join(pdb_file))

        # setup engine
        self.engine = ops_lammps.Engine(
            inputs="\n".join(int_system),
            lmp_cmdargs=['-echo', 'none', '-screen', 'none'],
            options={'n_steps_per_frame': 2,
                     'n_frames_max': 5},
            topology=self.pdb_file
        )

    def teardown(self):
        if os.path.exists(self.pdb_file):
            os.remove(self.pdb_file)
        if os.path.exists('log.lammps'):
            os.remove('log.lammps')

    def test_sanity(self):
        assert_equal(self.engine.n_steps_per_frame, 2)
        assert_equal(self.engine.n_frames_max, 5)
        # TODO: add more sanity checks
        pass  # not quite a SkipTest, but a reminder to add more

    def test_units_current_snapshot(self):
        """Test if we handle units correctly"""
        snap = self.engine._get_snapshot()
        template = ops_lammps.snapshot_from_pdb(self.pdb_file)
        assert_equal_array_array(snap.coordinates, template.coordinates)