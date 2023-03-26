from dolfinx.io import gmshio

import gmsh
import math
import random
import numpy as np


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

    length = 1.0
    diameter = 0.5
    h = 1.0 / N

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    model = gmsh.model()
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, h))
    
    if comm.rank == rank:
        pipe = model.occ.addCylinder(0, 0, 0, 0, length, 0, diameter / 2)

        model.occ.synchronize()

        walls = {
            'left': -1,
            'right': -1,
            'mid': -1
        }
        for dim, tag in gmsh.model.getEntities(dim=2):
            if np.allclose(gmsh.model.occ.getCenterOfMass(dim, tag), 
                           [0.0, 0.0, 0.0]):
                walls['left'] = tag
            elif np.allclose(gmsh.model.occ.getCenterOfMass(dim, tag), 
                             [0.0, 5.0, 0.0]):
                walls['right'] = tag
            else:
                walls['mid'] = tag

        # Marking physical boundaries
        
        model.addPhysicalGroup(2, [walls['left']], 1, 'left')
        model.addPhysicalGroup(2, [walls['right']], 2, 'right')
        model.addPhysicalGroup(2, [walls['mid']], 3, 'mid')
        model.addPhysicalGroup(3, [pipe], 1, 'pipe')

        model.mesh.generate(3)

    msh, mt, ft = gmshio.model_to_mesh(model, comm, rank)
    msh.name = "3d_pipe_type_0"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft
