import numpy as np

from dolfinx import fem, mesh, geometry
from ufl import Measure, inner


class BoundaryCondition():
    """
    This class represent generalized version of boundary conditions
    """
    def __init__(self, type, marker, values, V, u, v, measure):
        """
        Creates BoundaryConditon

        Parameters
        ----------
        type : {'Dirichlet', 'Neumann', 'Robin'}
            Type of boundary condition
        marker : int
            Boundary index
        values : array_like or callable
            Values applied to boundary
        V : dolfinx.fem.FunctionSpace
            FunctionSpace or VectorFunctionSpace for a given problem
        u : ufl.Argument
            Trial function argument
        v : ufl.Argument
            Test function argument
        measure : dict
            Disctionary returned by `generate_boundary_measure` function
        Raises
        ------
        TypeError
            Raises when unknown type passed as a first argument
        """

        ds = measure['Measure']
        fdim, facets, facet_tag = measure['Facets']
        
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(V)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)            
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            self._bc = fem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError(f"Unknown boundary condition: {type}")
    
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


def generate_boundary_measure(boundaries, domain):
    """
    Boundary measure generation

    Parameters
    ----------
    boundaries : list of tuples
        Each tuple represent specific boundary. 
            First parameter is boundary index(int).   
            Second - function from `x` that describe the boundary.
    domain : dolfinx.mesh.Mesh
        Mesh for a given problem

    Returns
    -------
    dict
        Returns required parameters to set up BoundaryCondition
    """    

    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    return {
        'Measure': ds,
        'Facets': (fdim, facets, facet_tag)
    }

def eval_pointvalues(uh, x, distance_tolerance=1e-15):
    """
    Interpolation of dolfinx.Function at specified point.    
    Source: https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html

    Parameters
    ----------
    uh
        function from which quantities spud be interpolated
    x
        spacial coordinates of points (ndarray, size: nPoints x dimSpace)
    distance_tolerance : float, default=1e-15
        Tolerance for how close an entity has to be to consider it a collision

    Returns
    -------
    ndarray
        Interpolated values at specified points (ndarray, size: nPoints x dimFuncSpace)
    """    

    # Get domain
    domain = uh.function_space.mesh

    # Create searchable data structure of current mesh
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []
    
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, x)
    
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, x)
    for i, point in enumerate(x):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    # Interpolate values at points on current processor
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    values = uh.eval(points_on_proc, cells)

    return values

def project(expression, domain, element):
    """
    Project expression on FunctionSpace

    Parameters
    ----------
    expression
    domain : ufl.domain
    space : ufl.element

    Returns
    -------
    dolfinx.fem.Function
    """
    
    V = fem.FunctionSpace(domain, element)
    expr = fem.Expression(expression, V.element.interpolation_points)
    fun = fem.Function(V)
    fun.interpolate(expr)

    return fun
