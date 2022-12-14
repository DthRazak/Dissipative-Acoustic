#!/usr/bin/env python3

from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import (TrialFunction, TestFunction, dx, grad, div, inner, lhs, rhs)

from dolfinx.common import Timer, TimingType, list_timings

import argparse
import numpy as np
import pandas as pd

from utils.dolfinx import (BoundaryCondition, generate_boundary_measure, 
                           project, L2_norm, H1_norm)

from pathlib import Path


save_dir = "./results/1dh-non-stationary"
Path(save_dir).mkdir(parents=True, exist_ok=True)


def generate_boundaries(points):
    return [(1, lambda x: np.isclose(x[0], points[0])), 
            (2, lambda x: np.isclose(x[0], points[1]))]

def problem_setup(N, points, p, T, t, ct, dt, fluid, solver, pc, prefix):
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
        
    # Construction of bilinear form and linear functional
    GAMMA, BETA = 0.5, 0.5
    
    mm = ScalarType(ro) * inner(u, v) * dx
    aa = ScalarType(ro * c**2) * inner(grad(u), grad(v)) * dx
    cc = ScalarType(4./3 * eta) * inner(grad(u), grad(v)) * dx
    
    F = mm + (dt * GAMMA * cc) + (0.5 * dt * dt * BETA * aa)
        
    boundaries = generate_boundaries(points)
    measure = generate_boundary_measure(boundaries, domain)
    
    u_D = lambda x: x[0] * 0.0
    u_N1 = fem.Constant(domain, ScalarType(-p))
    u_N2 = fem.Constant(domain, ScalarType(0.0))
    
    bcs = [BoundaryCondition("Dirichlet", 1, u_D, V, u, v, measure).bc]
    nbcs = [BoundaryCondition("Neumann", 2, u_N1, V, u, v, measure).bc,
            BoundaryCondition("Neumann", 2, u_N2, V, u, v, measure).bc] 
    
    return {
        'Params': (N, points, fluid, ro, c, T, dt, t, ct, GAMMA, BETA, solver, pc, prefix),
        'FunctionSpace': (domain, V, u, v),
        'Forms': (mm, aa, cc), 
        'Problem': (F, bcs, nbcs)
    }


def solve_problem(config):
    N, points, fluid, ro, c, T, dt, t_stop, ct, GAMMA, BETA, solver_type, pc, prefix = config['Params']
    domain, V, u, v = config['FunctionSpace']
    mm, aa, cc = config['Forms']
    F, bcs, nbcs = config['Problem']
    
    if prefix:
        PREFIX = f'{solver_type}_{pc}_{MPI.COMM_WORLD.size}_'
    else:
        PREFIX = ''

    U_PATH = f"{save_dir}/{PREFIX}uj.xdmf"
    UD_PATH = f"{save_dir}/{PREFIX}uj_d.xdmf"
    PRESSURE_PATH = f"{save_dir}/{PREFIX}pressure.xdmf"

    xdmf_u = io.XDMFFile(domain.comm, U_PATH, "w")
    xdmf_u_d = io.XDMFFile(domain.comm, UD_PATH, "w")
    xdmf_p = io.XDMFFile(domain.comm, PRESSURE_PATH, "w")
    xdmf_u.write_mesh(domain)
    xdmf_u_d.write_mesh(domain)
    xdmf_p.write_mesh(domain)

    # Create initial condition
    initial_condition = lambda x: x[0] * 0.0
    
    uj = fem.Function(V)
    uj_d = fem.Function(V)
    uj_dd = fem.Function(V)
    uj_d_d_tg = fem.Function(V)
    
    uj.interpolate(initial_condition)
    uj_d.interpolate(initial_condition)
    uj_dd.interpolate(initial_condition)
    uj_d_d_tg.interpolate(initial_condition)
    
    p = project(ro * c**2 * uj.dx(0), domain, ('CG', 1))
    p.x.array[:] = ro * c**2 * uj_d.x.array
    
    # Construct the left and right hand side of the problem
    bilinear_form = fem.form(lhs(F))
    
    A = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(solver_type.lower())
    solver.getPC().setType(pc.lower())
    
    uj_d_d_tg.x.array[:] = dt * GAMMA * uj_d.x.array
        
    L1 = nbcs[0] + (cc * uj_d) + (aa * uj) + (aa * uj_d_d_tg)
    L2 = nbcs[1] + (cc * uj_d) + (aa * uj) + (aa * uj_d_d_tg)
    linear_form_1 = fem.form(rhs(L1))
    linear_form_2 = fem.form(rhs(L2))
    
    b1 = fem.petsc.create_vector(linear_form_1)
    b2 = fem.petsc.create_vector(linear_form_2)

    has_norm = False
    L2_n, H1_n = 0.0, 0.0

    # Solve problem at each time step,
    num_steps = int(T / dt)
    save_step = 1 if num_steps // 100 < 1 else num_steps // 100
    t = 0.0

    xdmf_u.write_function(uj, t)
    xdmf_u_d.write_function(uj_d, t)
    xdmf_p.write_function(p, t)

    for i in range(num_steps):
        t += dt
        
        uj_d_d_tg.x.array[:] = dt * GAMMA * uj_d.x.array
        
        b = b1 if t < t_stop else b2
        linear_form = linear_form_1 if t < t_stop else linear_form_2

        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)
        
        # Solve linear problem
        solver.solve(b, uj_dd.vector)
        uj_dd.x.scatter_forward()
        
        # Update solution
        uj.x.array[:] = uj.x.array + dt * uj_d.x.array + 0.5 * dt**2 * uj_dd.x.array
        uj_d.x.array[:] = uj_d.x.array + dt * uj_dd.x.array

        # Save results into the files
        if i % save_step == 0:
            p.x.array[:] = ro * c**2 * uj_d.x.array

            xdmf_u.write_function(uj, t)
            xdmf_u_d.write_function(uj_d, t)
            xdmf_p.write_function(p, t)

        if not has_norm and t >= ct:
            L2_n = L2_norm(uj)
            H1_n = H1_norm(uj)

            has_norm = True

    xdmf_u.close()
    xdmf_u_d.close()
    xdmf_p.close()
        
    return L2_n, H1_n

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, help="Number of finite elements",
                        metavar='N', default=500)
    parser.add_argument('--dt', type=float, help="Time integration step",
                        metavar='dt', default=5.0e-8)
    parser.add_argument('--t', type=float, help="Pressure stop time",
                        metavar='t', default=1.5e-6)
    parser.add_argument('--T', type=float, help="Stop time",
                        metavar='T', default=5.0e-5)
    parser.add_argument('--ct', type=float, help="Control time",
                        metavar='ct', default=2.0e-5)
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
        points=[0.0, 0.02],
        p=1.0,
        T=args.T,
        t=args.t,
        ct=args.ct,
        dt=args.dt,
        fluid=args.fluid,
        solver=args.solver,
        pc=args.pc,
        prefix=args.prefix
    )

    L2_n, H1_n = solve_problem(config)

    # list_timings(MPI.COMM_WORLD, [TimingType.wall])

    if MPI.COMM_WORLD.rank == 0:
        print(f'L2 norm = {L2_n}')
        print(f'H1 norm = {H1_n}')


if __name__ == "__main__":
    main()
