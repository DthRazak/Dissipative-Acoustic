#!/usr/bin/env python3

from urllib.request import ftpwrapper
from dolfinx import geometry, fem, mesh, plot, io
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType, ComplexType
from ufl import (TrialFunction, Measure, TestFunction, dx, ds, grad, div, inner, lhs, rhs)

from dolfinx.common import Timer, TimingType, list_timings

import argparse
import numpy as np
import pandas as pd

from utils.dolfinx import (BoundaryCondition, generate_boundary_measure, 
                           project, L2_norm, H1_norm)
from utils.meshing import create_square_mesh

from pathlib import Path


save_dir = "./results/2dh"
Path(save_dir).mkdir(parents=True, exist_ok=True)


def problem_setup(N, points, p, freq, fluid, solver, pc, prefix):    
    """
    Performs problem configuration w.r.t. given parameters
    """
    
    # Mesh and function space definition
    domain, mt, ft = create_square_mesh(points[0][1], N, MPI.COMM_WORLD, 0)

    V = fem.VectorFunctionSpace(domain, ("CG", 2))

    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Definition functions of physical characteristics
    fluids = pd.read_csv('../../data/physical_properties.csv', sep=';', index_col='Fluid')
    
    ro, c, eta = fluids.loc[fluid, ['Density', 'Speed of sound', 'Viscosity']]
    omega = freq * 2 * np.pi
        
    # Construction of problem form
    GAMMA, BETA = 0.5, 0.5
    
    mm = ScalarType(ro) * inner(u, v) * dx
    aa = ScalarType(ro * c**2) * inner(grad(u), grad(v)) * dx
    cc = ScalarType(4./3 * eta) * inner(grad(u), grad(v)) * dx
    
    F = aa - omega**2 * mm + omega * 1.0j * cc
    
    # Definition of boundary conditions
    measure = generate_boundary_measure([], domain, ft)
    
    u_D = lambda x: [x[0] * 0.0, x[1] * 0.0, x[2] * 0.0]
    u_N = fem.Constant(domain, ScalarType(p))

    bcs = [BoundaryCondition("Dirichlet", 1, u_D, V, u, v, measure).bc,
           BoundaryCondition("Dirichlet", 2, u_D, V, u, v, measure).bc, 
           BoundaryCondition("Dirichlet", 3, u_D, V, u, v, measure).bc]
     
    nbcs = [BoundaryCondition("Neumann", 4, u_N, V, u, v, measure).bc] 
    
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

    PREFIX = f'{solver_type}_{pc}_{MPI.COMM_WORLD.size}'

    U_PATH = f"{save_dir}/{PREFIX}_uh.xdmf"
    PRESSURE_PATH = f"{save_dir}/{PREFIX}pressure.xdmf"

    xdmf_u = io.XDMFFile(domain.comm, U_PATH, "w")
    xdmf_p = io.XDMFFile(domain.comm, PRESSURE_PATH, "w")
    xdmf_u.write_mesh(domain)
    xdmf_p.write_mesh(domain)

    petsc_options = {"ksp_type": solver_type.lower(), "pc_type": pc.lower()}

    problem = fem.petsc.LinearProblem(lhs(F), rhs(F), bcs=bcs, petsc_options=petsc_options)
    uh = problem.solve()

    p = project(ro * c**2 * div(uh), domain, ("CG", 1))

    xdmf_u.write_function(uh)
    xdmf_p.write_function(p)
        
    xdmf_u.close()
    xdmf_p.close()
    
    # Norm calculation
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
                        metavar='N', default=50)
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
        points=[[0.0, 0.002], [0.0, 0.002]],
        p=(-1.0, 0.0, 0.0),
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
