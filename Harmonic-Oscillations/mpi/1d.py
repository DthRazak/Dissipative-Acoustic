#!/usr/bin/env python3

from distutils.sysconfig import PREFIX
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType 
from ufl import (TrialFunction, TestFunction, dx, ds, grad, inner, lhs, rhs)

from dolfinx.common import Timer, TimingType, list_timings

import argparse
import numpy as np
import pandas as pd

from utils.dolfinx import (BoundaryCondition, generate_boundary_measure, 
                           project, L2_norm, H1_norm)

from pathlib import Path


save_dir = "./results/1dh"
Path(save_dir).mkdir(parents=True, exist_ok=True)


def generate_boundaries(points):
    return [(1, lambda x: np.isclose(x[0], points[0])), 
            (2, lambda x: np.isclose(x[0], points[1]))]

def problem_setup(N, points, p, freq, fluid, solver, pc, prefix):
    """
    Performs problem configuration w.r.t. given parameters
    """
    
    # Mesh and function space definition
    domain = mesh.create_interval(MPI.COMM_WORLD, N, points)
    V = fem.FunctionSpace(domain, ("CG", 1))

    u = TrialFunction(V)
    v = TestFunction(V)

    # Definition of density and speed functions
    fluids = pd.read_csv('../../data/physical_properties.csv', sep=';', index_col='Fluid')

    ro, c, eta = fluids.loc[fluid, ['Density', 'Speed of sound', 'Viscosity']]
    omega = freq * 2 * np.pi
        
    # Construction of bilinear form and linear functional
    aa = ScalarType(ro * c**2) * inner(grad(u), grad(v)) * dx
    mm = ScalarType(ro) * inner(u, v) * dx
    cc = ScalarType(4./3 * eta) * inner(grad(u), grad(v)) * dx

    F = aa - omega**2 * mm + omega * 1.0j * cc
    
    boundaries = generate_boundaries(points)
    measure = generate_boundary_measure(boundaries, domain)

    u_D = lambda x: x[0] * 0.0
    u_N = fem.Constant(domain, ScalarType(-p))

    bcs = [BoundaryCondition("Dirichlet", 1, u_D, V, u, v, measure).bc]
    nbcs = [BoundaryCondition("Neumann", 2, u_N, V, u, v, measure).bc] 

    return {
        'Params': (N, points, p, ro, c, omega, fluid, solver, pc, prefix),
        'FunctionSpace': (domain, V, u, v),
        'Problem': (F, bcs, nbcs)
    }

def solve_problem(config):
    N, points, p, ro, c, omega, fluid, solver_type, pc, prefix = config['Params']
    domain, V, u, v = config['FunctionSpace']
    F, bcs, nbcs = config['Problem']

    F += nbcs[0]

    if prefix:
        PREFIX = f'{solver_type}_{pc}_{MPI.COMM_WORLD.size}_'
    else:
        PREFIX = ''

    U_PATH = f"{save_dir}/{PREFIX}uh.xdmf"
    PRESSURE_PATH = f"{save_dir}/{PREFIX}pressure.xdmf"

    xdmf_u = io.XDMFFile(domain.comm, U_PATH, "w")
    xdmf_p = io.XDMFFile(domain.comm, PRESSURE_PATH, "w")
    xdmf_u.write_mesh(domain)
    xdmf_p.write_mesh(domain)

    petsc_options = {"ksp_type": solver_type.lower(), "pc_type": pc.lower()}

    problem = fem.petsc.LinearProblem(lhs(F), rhs(F), bcs=bcs, petsc_options=petsc_options)
    uh = problem.solve()
    uh.x.scatter_forward()

    p = project(ro * c**2 * uh.dx(0), domain, ('CG', 1))

    xdmf_u.write_function(uh)
    xdmf_p.write_function(p)
        
    xdmf_u.close()
    xdmf_p.close()

    # Norm computation
    uh_array = np.copy(uh.x.array[:])

    uh.x.array[:] = np.zeros(uh_array.shape, dtype=uh_array.dtype)
    uh.x.array.real[:] = uh_array.real
    L2_n_r = L2_norm(uh)
    H1_n_r = H1_norm(uh)

    uh.x.array[:] = np.zeros(uh_array.shape, dtype=uh_array.dtype)
    uh.x.array.real[:] = uh_array.imag
    L2_n_c = L2_norm(uh)
    H1_n_c = H1_norm(uh)

    return L2_n_r, H1_n_r, L2_n_c, H1_n_c

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, help="Number of finite elements",
                        metavar='N', default=500)
    parser.add_argument('--freq', type=float, help="Frequency",
                        metavar='freq', default=5.0e5)
    parser.add_argument('--fluid', type=str, help="Fluid name",
                        metavar='fluid', default="Water")
    parser.add_argument('--solver', type=str, help="Solver Type",
                        metavar='solver', default="CG")
    parser.add_argument('--pc', type=str, help="Preconditioner Type",
                        metavar='pc', default="ASM")
    parser.add_argument('--prefix', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    config = problem_setup(
        N=args.n,
        points=[0.0, 0.005],
        p=1.0,
        freq=args.freq,
        fluid=args.fluid,
        solver=args.solver,
        pc=args.pc,
        prefix=args.prefix
    )

    L2_n_r, H1_n_r, L2_n_c, H1_n_c = solve_problem(config)

    # list_timings(MPI.COMM_WORLD, [TimingType.wall])

    if MPI.COMM_WORLD.rank == 0:
        print(f'L2(R) norm = {np.real(L2_n_r)}')
        print(f'H1(R) norm = {np.real(H1_n_r)}')
        # Complex values is also stored in real part
        print(f'L2(C) norm = {np.real(L2_n_c)}')
        print(f'H1(C) norm = {np.real(H1_n_c)}')


if __name__ == "__main__":
    main()
