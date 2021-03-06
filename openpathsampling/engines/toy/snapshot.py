"""

@author: JD Chodera
@author: JH Prinz
"""

from openpathsampling.engines import BaseSnapshot, SnapshotFactory
import openpathsampling.engines.features as feats
from . import features as toy_feats


@feats.attach_features([
    feats.velocities,
    feats.coordinates,
    toy_feats.instantaneous_temperature,
    feats.engine
])
class ToySnapshot(BaseSnapshot):
    """
    Simulation snapshot. Only references to coordinates and velocities
    """

    @property
    def topology(self):
        return self.engine.topology

    @property
    def masses(self):
        return self.topology.masses


# The following code does almost the same as above

# ToySnapshot = SnapshotFactory(
#     name='ToySnapshot',
#     features=[
#         features.velocities,
#         features.coordinates,
#         features.engine
#     ],
#     description="Simulation snapshot. Only references to coordinates and "
#                 "velocities",
#     base_class=BaseSnapshot
# )
