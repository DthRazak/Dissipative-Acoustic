import gmsh
import numpy as np

from dolfinx.io import gmshio
from .segment_builder import SegmentBuilder


# ---------------------------------------------------------------
#                    Build Mesh Function
# ---------------------------------------------------------------
def build_mesh(config, comm, rank, show_logs):
    """_summary_

    Parameters
    ----------
    config: dict
        Configuration parameters from `config['pipe']`
    comm : _MPI.Comm
        The MPI communicator to use for mesh creation
    rank : int
        The rank the Gmsh model is initialized on
    show_logs : bool
        Indicates if there is need to print logging information
    
    Returns
    -------
        A triplet (mesh, cell_tags, facet_tags) where cell_tags hold
        markers for the cells, facet tags holds markers for facets
        if found in Gmsh model.
    """

    log_level = 1 if show_logs else 0
    pipe_handlers = {
        1: _build_pipe_1
    }

    return pipe_handlers[config['type']](config, comm, rank, log_level)

# ---------------------------------------------------------------
#                    Pipe 1 Build Function
# ---------------------------------------------------------------
def _build_pipe_1(config, comm, rank, log_level):
    h = 1.0 / config['N']
    segmentBuilder = SegmentBuilder()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", log_level)
    
    model = gmsh.model()
    model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, h))
    
    if comm.rank == rank:
        # Points defining boundaries of two connected `type 1` segments
        #  p2---------------------p3
        #   |    1     |    2     |
        #  p1---------------------p4
        p1 = model.occ.addPoint(0.0, 0.0, 0.0)
        p2 = model.occ.addPoint(0.0, segmentBuilder.diameter, 0.0)
        p3 = model.occ.addPoint(2 * segmentBuilder.length, segmentBuilder.diameter, 0.0)
        p4 = model.occ.addPoint(2 * segmentBuilder.length, 0.0, 0.0)

        l1 = model.occ.addLine(p2, p3)
        l2 = model.occ.addLine(p1, p4)
        l3 = model.occ.addLine(p1, p2)
        l4 = model.occ.addLine(p3, p4)

        boundary_curve = model.occ.addCurveLoop([l1, l4, l2, l3])

        # Generating bubbles for 1-st segment of type 1
        bubble_centres = segmentBuilder.build(1,
                                              0.0,
                                              0.0, 
                                              config['bubble_radius'], 
                                              config['bubble_lvl'][0]).tolist()
        
        # Generating bubbles for 2-nd segment of type 1
        bubble_centres = bubble_centres + \
                         segmentBuilder.build(1,
                                              segmentBuilder.length, 
                                              0.0,
                                              config['bubble_radius'], 
                                              config['bubble_lvl'][1]).tolist()
        
        # Meshing bubbles inside pipe
        bubbles = []
        for x, y in bubble_centres:
            circle = model.occ.addCircle(x, y, 0.0, config['bubble_radius'])
            bubble = model.occ.addCurveLoop([circle])
            bubbles.append(model.occ.addPlaneSurface([bubble]))

        s1 = model.occ.addPlaneSurface([boundary_curve] + bubbles)

        model.occ.synchronize()

        # Marking physical boundaries
        model.addPhysicalGroup(1, [l3], 1, 'left')
        model.addPhysicalGroup(1, [l4], 2, 'right')
        model.addPhysicalGroup(1, [l1, l2], 3, 'wall')

        model.addPhysicalGroup(2, [s1], 1, 'plane')
        model.addPhysicalGroup(2, bubbles[:-1], 2, 'bubbles')
        
        model.mesh.generate(2)

    msh, mt, ft = gmshio.model_to_mesh(model, comm, rank)
    msh.name = "2d_pipe_type_0"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft
