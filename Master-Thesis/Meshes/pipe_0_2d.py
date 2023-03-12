from dolfinx.io import gmshio

import gmsh
import random


def build_normal_mesh(N, comm, rank):
    """
    Create mesh for `Type 0` Pipe.
        `Length`:       5 m.
        `Diameter`:     0.5 m.

    Parameters
    ----------
    N : int
        Number of finite elements per 1 m.
    comm : _MPI.Comm
        The MPI communicator to use for mesh creation
    rank : int
        The rank the Gmsh model is initialized on

    Returns
    -------
        A triplet (mesh, cell_tags, facet_tags) where cell_tags hold
        markers for the cells, facet tags holds markers for facets
        if found in Gmsh model.
    """

    length = 5.0
    diameter = 0.5
    h = 1.0 / N

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    model = gmsh.model()
    
    if comm.rank == rank:
        p1 = model.occ.addPoint(0.0, 0.0, 0.0, h)
        p2 = model.occ.addPoint(0.0, diameter, 0.0, h)
        p3 = model.occ.addPoint(length, diameter, 0.0, h)
        p4 = model.occ.addPoint(length, 0.0, 0.0, h)

        l1 = model.occ.addLine(p2, p3)
        l2 = model.occ.addLine(p1, p4)
        l3 = model.occ.addLine(p1, p2)
        l4 = model.occ.addLine(p3, p4)

        c1 = model.occ.addCurveLoop([l1, l4, l2, l3])

        s1 = model.occ.addPlaneSurface([c1])

        model.occ.synchronize()

        # Marking physical boundaries
        model.addPhysicalGroup(1, [l1], 1, 'top')
        model.addPhysicalGroup(1, [l2], 2, 'bottom')
        model.addPhysicalGroup(1, [l3], 3, 'left')
        model.addPhysicalGroup(1, [l4], 4, 'right')

        model.addPhysicalGroup(2, [s1], 1, 'plane')

        model.mesh.generate(2)

    msh, mt, ft = gmshio.model_to_mesh(model, comm, rank)
    msh.name = "2d_pipe_type_0"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft

def build_bubble_mesh(N, bubbles_lvl, comm, rank):
    """
    Create mesh for `Type 0` Pipe with randomly generated bubles.
        `Length`:       5 m.
        `Diameter`:     0.5 m.
        `BubbleRadius`: 0.025 m.

    Parameters
    ----------
    N : int
        Number of finite elements per 1 m.
    bubbles_lvl : float
        Percentage of bubbles in pipe ranging from 0 to 0.5
    comm : _MPI.Comm
        The MPI communicator to use for mesh creation
    rank : int
        The rank the Gmsh model is initialized on

    Returns
    -------
        A triplet (mesh, cell_tags, facet_tags) where cell_tags hold
        markers for the cells, facet tags holds markers for facets
        if found in Gmsh model.
    """

    length = 5.0
    diameter = 0.5
    h = 1.0 / N
    bubble_radius = 0.025
    bubbles_lvl = max(min(bubbles_lvl, 0.5), 0.0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    model = gmsh.model()
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, h))
    
    if comm.rank == rank:
        # Generating base model of pipe
        p1 = model.occ.addPoint(0.0, 0.0, 0.0)
        p2 = model.occ.addPoint(0.0, diameter, 0.0)
        p3 = model.occ.addPoint(length, diameter, 0.0)
        p4 = model.occ.addPoint(length, 0.0, 0.0)

        l1 = model.occ.addLine(p2, p3)
        l2 = model.occ.addLine(p1, p4)
        l3 = model.occ.addLine(p1, p2)
        l4 = model.occ.addLine(p3, p4)

        c1 = model.occ.addCurveLoop([l1, l4, l2, l3])

        s1 = model.occ.addPlaneSurface([c1])

        # Generating random bubbles inside pipe
        max_num_of_bubbles = length / (2 * bubble_radius) * \
                             diameter / (2 * bubble_radius)
        bubles = list()
        for i in range(int(bubbles_lvl * max_num_of_bubbles)):
            x = random.uniform(0.0 + bubble_radius, length - bubble_radius)
            y = random.uniform(0.0 + bubble_radius, diameter - bubble_radius)

            circle = model.occ.addCircle(x, y, 0.0, bubble_radius)
            bubble = model.occ.addCurveLoop([circle])
            bubles.append(model.occ.addPlaneSurface([bubble]))

        model.occ.synchronize()

        # Marking physical boundaries
        model.addPhysicalGroup(1, [l1], 1, 'top')
        model.addPhysicalGroup(1, [l2], 2, 'bottom')
        model.addPhysicalGroup(1, [l3], 3, 'left')
        model.addPhysicalGroup(1, [l4], 4, 'right')

        model.addPhysicalGroup(2, [s1], 5, 'plane')
        model.addPhysicalGroup(2, bubles, 6, 'bubbles')
        
        model.mesh.generate(2)

    msh, mt, ft = gmshio.model_to_mesh(model, comm, rank)
    msh.name = "2d_pipe_type_0"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft
