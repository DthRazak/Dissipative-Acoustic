import gmsh

from dolfinx.io import XDMFFile, gmshio


def create_square_mesh(a, N, comm, rank):
    """
    Create square mesh with triangulation

    Parameters
    ----------
    a : float
        Square size
    N : int
        Number of finite elements on the boundary
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

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    model = gmsh.model()
    
    if comm.rank == rank:
        h = a / N

        p1 = model.occ.addPoint(0.0, 0.0, 0.0, h)
        p2 = model.occ.addPoint(0.0, a, 0.0, h)
        p3 = model.occ.addPoint(a, 0.0, 0.0, h)
        p4 = model.occ.addPoint(a, a, 0.0, h)

        l1 = model.occ.addLine(p1, p3)
        l2 = model.occ.addLine(p1, p2)
        l3 = model.occ.addLine(p2, p4)
        l4 = model.occ.addLine(p4, p3)

        c1 = model.occ.addCurveLoop([l1, l2, l3, l4])

        s1 = model.occ.addPlaneSurface([c1])

        model.occ.synchronize()

        # Marking physical boundaries
        model.addPhysicalGroup(1, [l1], 1)  # bottom
        model.addPhysicalGroup(1, [l2], 2)  # left
        model.addPhysicalGroup(1, [l3], 3)  # top
        model.addPhysicalGroup(1, [l4], 4)  # right

        model.addPhysicalGroup(2, [s1], 5)  # plane

        model.mesh.generate(2)

    msh, mt, ft = gmshio.model_to_mesh(model, comm, rank)
    msh.name = "rect"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    gmsh.finalize()

    return msh, mt, ft

def mesh_to_XDMFFile(mesh, cell_tags, facet_tags, filename='mesh.xdmf'):
    """
    Save dolfinx mesh to XDMFFile

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        
    cell_tags : dolfinx.cpp.mesh.MeshTags
        
    facet_tags : dolfinx.cpp.mesh.MeshTags
        
    filename : str, optional
        Path to save, by default 'mesh.xdmf'
    """    

    with XDMFFile(mesh.comm, filename, "w") as file:
        file.write_mesh(mesh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        file.write_meshtags(cell_tags, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")
        file.write_meshtags(facet_tags, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")
