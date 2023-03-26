import gmsh
import numpy as np

from dolfinx.io import gmshio
from .segment_builder import SegmentBuilder


# ---------------------------------------------------------------
#                    Build Mesh Function
# ---------------------------------------------------------------
def build_mesh(config, comm, rank, show_logs):
    """
    Build rectangle mesh and fill its segments with bubbles.

    Parameters
    ----------
    config: dict
        Configuration parameters from `config['mesh']`
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

    h = 1.0 / config['N']
    segmentBuilder = SegmentBuilder(0.5, 0.5)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", log_level)
    
    model = gmsh.model()
    model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, h))
    
    if comm.rank == rank:
        # Points defining boundaries of ten connected `type 1` segments
        #  p2-------------------p3
        #   | | | | | | | | | | |
        #  p1-------------------p4
        p1 = model.occ.addPoint(0.0, 0.0, 0.0)
        p2 = model.occ.addPoint(0.0, segmentBuilder.diameter, 0.0)
        p3 = model.occ.addPoint(10 * segmentBuilder.length, segmentBuilder.diameter, 0.0)
        p4 = model.occ.addPoint(10 * segmentBuilder.length, 0.0, 0.0)

        l1 = model.occ.addLine(p2, p3)
        l2 = model.occ.addLine(p1, p4)
        l3 = model.occ.addLine(p1, p2)
        l4 = model.occ.addLine(p3, p4)

        boundary_curve = model.occ.addCurveLoop([l1, l4, l2, l3])

        # Meshing bubbles inside segments
        bubbles = []
        for x, y in config['bubble_centres']:
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
    msh.name = "2d_rect_type_0"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft


# ---------------------------------------------------------------
#                    Build Mesh Function
# ---------------------------------------------------------------
def generate_bubbles(bubble_radius, bubble_lvl):
    """
    Generate bubbles cetrhes for list of segments.

    Parameters
    ----------
    bubble_radius : float
        Bubble radius
    bubble_lvl : float, 0.0 to 0.5
        Contamination percentage of segment

    Returns
    -------
        Tuple of `bubble_centres` and real contamination level 
        `bubble_lvl_real`.
    """
    segmentBuilder = SegmentBuilder(0.5, 0.5)
    segment_sq = segmentBuilder.diameter * segmentBuilder.length

    bubble_centres = []
    bubble_lvl_real = []
    for i in range(len(bubble_lvl)):
        segment_centres = segmentBuilder.build(1,
            i * segmentBuilder.length, 
            0.0,
            bubble_radius, 
            bubble_lvl[i]).tolist()
        bubble_lvl_real.append(
            (len(segment_centres) * np.pi * bubble_radius ** 2) / segment_sq)
        bubble_centres = bubble_centres + segment_centres

    bubble_lvl_real = np.round(bubble_lvl_real, 4)

    return bubble_centres, bubble_lvl_real
