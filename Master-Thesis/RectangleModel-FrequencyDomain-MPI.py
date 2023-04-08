#!/usr/bin/env python3

from dolfinx import fem, io
from dolfinx.common import Timer, TimingType, list_timings
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import (TrialFunction, Measure, TestFunction, grad, div, inner, lhs, rhs)

import random
import numpy as np
import pandas as pd

from utils.dolfinx import BoundaryCondition, generate_boundary_measure, project
from meshes import rectangle_builder

from pathlib import Path

import argparse

frequency_save_dir = "/root/Meshes/rectangle-model/frequency"
Path(frequency_save_dir).mkdir(parents=True, exist_ok=True)

def frequency_domain_problem_setup(config, mesh_data):    
    """
    Perform problem configuration of frequency domain problem
    w.r.t. given parameters. Add new information to config.
    
    Parameters
    ----------
    config : dict
        Configuration parameters required for problem setup
    mesh_data: tuple
        A triplet (mesh, cell_tags, facet_tags) of Gmsh model
    """
    
    # ---------------------------------------------------------------
    #               Mesh and function space definition
    # ---------------------------------------------------------------
    domain, mt, ft = mesh_data
    dx = Measure('dx', subdomain_data=mt, domain=domain)
    ds = Measure("ds", subdomain_data=ft, domain=domain)
    
    V = fem.VectorFunctionSpace(domain, ("CG", 2))

    u = TrialFunction(V)
    v = TestFunction(V)

    # ---------------------------------------------------------------
    #        Definition physical characteristics functions
    # ---------------------------------------------------------------
    fluids = pd.read_csv('../data/physical_properties.csv', sep=';', index_col='Fluid')
    
    ro, c, eta = fluids.loc[[config['problem']['fluid'], 
                             config['problem']['contaminant']], 
                            ['Density', 'Speed of sound', 'Viscosity']].T.values
    omega = config['problem']['control_frequencies'][config['problem']['freq_idx']] * 2 * np.pi

    # ---------------------------------------------------------------
    #               Construction of problem form
    # ---------------------------------------------------------------
    GAMMA, BETA = 0.5, 0.5

    mm = ScalarType(ro[0]) * inner(u, v) * dx(1)
    aa = ScalarType(ro[0] * c[0]**2) * inner(grad(u), grad(v)) * dx(1)
    cc = ScalarType(4./3 * eta[0]) * inner(grad(u), grad(v)) * dx(1)
    
    if len(config['mesh']['bubble_centres']) > 0:
        mm += ScalarType(ro[1]) * inner(u, v) * dx(2)
        aa += ScalarType(ro[1] * c[1]**2) * inner(grad(u), grad(v)) * dx(2)
        cc += ScalarType(4./3 * eta[1]) * inner(grad(u), grad(v)) * dx(2)
    
    F = aa - omega**2 * mm + omega * 1.0j * cc
    
    # ---------------------------------------------------------------
    #               Definition of boundary conditions
    # ---------------------------------------------------------------
    measure = generate_boundary_measure([], domain, ft)
    
    u_D = lambda x: [x[0] * 0.0, x[1] * 0.0, x[2] * 0.0]
    u_N = fem.Constant(domain, ScalarType((config['problem']['pressure'], 0, 0)))

    bcs = [BoundaryCondition("Dirichlet", 3, u_D, V, u, v, measure).bc,
           BoundaryCondition("Dirichlet", 2, u_D, V, u, v, measure).bc]
     
    nbcs = [BoundaryCondition("Neumann", 1, u_N, V, u, v, measure).bc] 
    
    # ---------------------------------------------------------------
    #                      Update config
    # ---------------------------------------------------------------
    config['problem']['function_space'] = {
        'domain': domain,
        'V': V,
        'u': u,
        'v': v,
        'dx': dx,
        'ds': ds
    }
    config['problem']['physical_properties'] = {
        'ro': ro, 
        'c': c, 
        'eta': eta,
        'omega': omega
    }
    config['problem']['forms'] = {
        'M': mm,
        'A': aa,
        'C': cc,
        'F': F
    }
    config['problem']['boundary_conditions'] = {
        'Dirichlet': bcs,
        'Neumann': nbcs
    }

def solve_frequency_domain_problem(config):
    # ----------------------------------------------------------------
    #                   Obtaining the required data
    # ----------------------------------------------------------------

    domain, V, ds, dx = config['problem']['function_space']['domain'], \
                        config['problem']['function_space']['V'], \
                        config['problem']['function_space']['ds'], \
                        config['problem']['function_space']['dx']

    mm, aa, cc, F = config['problem']['forms']['M'], \
                    config['problem']['forms']['A'], \
                    config['problem']['forms']['C'], \
                    config['problem']['forms']['F']

    bcs, nbcs = config['problem']['boundary_conditions']['Dirichlet'], \
                config['problem']['boundary_conditions']['Neumann']
    
    ro, c, omega = config['problem']['physical_properties']['ro'], \
                   config['problem']['physical_properties']['c'], \
                   config['problem']['physical_properties']['omega']

    # ----------------------------------------------------------------
    #              Configuring path for storing results
    # ----------------------------------------------------------------
    
    if config['results']['file_prefix']:
        PREFIX = f"N_{config['mesh']['N']}" \
                 f"_omega_{int(np.round(omega))}" \
                 f"_br_{str(config['mesh']['bubble_radius']).split('.')[1]}" \
                 f"_bl_{'_'.join(str(int(pct * 100)) for pct in config['mesh']['bubble_lvl'])}_"
    else:
        PREFIX = ''
    
    U_PATH = f"{frequency_save_dir}/{PREFIX}uj.xdmf"
    PRESSURE_PATH = f"{frequency_save_dir}/{PREFIX}pressure.xdmf"
        
    # ----------------------------------------------------------------
    #                      Setuping solver
    # ----------------------------------------------------------------
    petsc_options = {
        "ksp_type": config['petsc']['solver'], 
        "pc_type": config['petsc']['pc']
    }

    F += nbcs[0]
    
    problem = fem.petsc.LinearProblem(lhs(F), 
                                      rhs(F), 
                                      bcs=bcs, 
                                      petsc_options=petsc_options)
    
    # ----------------------------------------------------------------
    #                       Solving problem
    # ----------------------------------------------------------------
    
    if config['logs']:
        print(f'Info    : Solving problem - Start')
    uh = problem.solve()
    if config['logs']:
        print(f'Info    : Solving problem - Done.')
    
    # ----------------------------------------------------------------
    #              Calculating pressure at right boundary
    # ----------------------------------------------------------------
    
    p = project(div(uh), domain, ("CG", 2))
    pressure_array = np.copy(p.x.array)

    fluid_cells = dx.subdomain_data().find(1)
    p.x.array[:] = pressure_array[:] * ro[0] * c[0]**2
    
    if len(config['mesh']['bubble_centres']) > 0:
        contamination_cells = dx.subdomain_data().find(2)
        p.x.array[contamination_cells] = pressure_array[contamination_cells] * ro[1] * c[1]**2

    p_int = fem.assemble_scalar(fem.form(p * ds(2)))
    result = np.round(p_int, 8)

    # ------------------------------------------------------------
    #                 Save results into the files
    # ------------------------------------------------------------
    if config['results']['save_mesh_to_file']:
        xdmf_u = io.XDMFFile(domain.comm, U_PATH, "w")
        xdmf_p = io.XDMFFile(domain.comm, PRESSURE_PATH, "w")
        
        xdmf_u.write_mesh(domain)
        xdmf_p.write_mesh(domain)
        
        xdmf_u.write_function(uh)
        xdmf_p.write_function(p)
        
        xdmf_u.close()
        xdmf_p.close()
        
    return result

def generate_frequency_config(bubble_centres, bubble_lvl):
    frequency_config = {
        'mesh': {
            'N': 80,
            'bubble_radius': 0.05,
            'bubble_centres': bubble_centres,
            'bubble_lvl': bubble_lvl
        },
        'problem': {
            'fluid': 'Water',
            'contaminant': 'Fuel oil',
            'pressure': 1e3,
            'freq_idx': 0,
            'control_frequencies': [1.5e3, 2e3, 3e3, 8e3]
        },
        'petsc': {
            'solver': 'preonly',
            'pc': 'lu'
        },
        'results': {
            'save_mesh_to_file': False,
            'file_prefix': False,
        },
        'logs': False
    }
    return frequency_config

def read_input(filename):
    input_data = []
    with open(filename, 'r') as input_file:
        for line in input_file:
            cnt_lvl, points = line.strip().split('|')
            bubble_centres = []
            for point in points.split(';'):
                if point == '':
                    break
                bubble_centres.append([float(xy) for xy in point.split(',')])

            input_data.append([[float(lvl) for lvl in cnt_lvl.split(';')], 
                                bubble_centres])
            
    return input_data

def write_results(frequency_data, filename, write_header=True):
    with open(filename, 'a') as file:
        if write_header:
            file.write(';'.join(f'sg_{i+1}' for i in range(10)))
            file.write(';')
            file.write(';'.join(f'f_{i+1}_r;f_{i+1}_i' 
                                  for i in range(len(frequency_data[0][1]))))
            file.write('\n')
        for data in frequency_data:
            file.write(';'.join(str(c_lvl) for c_lvl in data[0]))
            file.write(';')
            file.write(';'.join(str(p_int) for p_int in data[1]))
            file.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=int, help="Current interation",
                        metavar='I', default=1000)
    args = parser.parse_args()

    frequency_data = []
    log_step = 5
    split_step = 200

    input_data = read_input('./input/input_s5_1000.txt')

    i = args.i
    if i % log_step == 0:
        if MPI.COMM_WORLD.rank == 0:
            print(f'Info    : Step: {i+1}')

    bubble_lvl, bubble_centres = input_data[i]
    
    frequency_config = generate_frequency_config(bubble_centres, bubble_lvl)
    mesh_data = rectangle_builder.build_mesh(frequency_config['mesh'], 
                                            MPI.COMM_WORLD, 
                                            0, 
                                            frequency_config['logs'])

    frequency_config['problem']['freq_idx'] = 0
    frequency_domain_problem_setup(frequency_config, mesh_data)
    r0 = solve_frequency_domain_problem(frequency_config)

    frequency_config['problem']['freq_idx'] = 1
    frequency_domain_problem_setup(frequency_config, mesh_data)
    r1 = solve_frequency_domain_problem(frequency_config)

    frequency_config['problem']['freq_idx'] = 2
    frequency_domain_problem_setup(frequency_config, mesh_data)
    r2 = solve_frequency_domain_problem(frequency_config)

    frequency_config['problem']['freq_idx'] = 3
    frequency_domain_problem_setup(frequency_config, mesh_data)
    r3 = solve_frequency_domain_problem(frequency_config)

    r0 = MPI.COMM_WORLD.reduce(r0)
    r1 = MPI.COMM_WORLD.reduce(r1)
    r2 = MPI.COMM_WORLD.reduce(r2)
    r3 = MPI.COMM_WORLD.reduce(r3)

    if MPI.COMM_WORLD.rank == 0:
        r0_str = f'{np.real(r0)};{np.imag(r0)}'
        r1_str = f'{np.real(r1)};{np.imag(r1)}'
        r2_str = f'{np.real(r2)};{np.imag(r2)}'
        r3_str = f'{np.real(r3)};{np.imag(r3)}'

        write_results([[bubble_lvl, [r0_str, r1_str, r2_str, r3_str]]], 
                        f'./results/frequency/res_fuel_oil_s5_1000_p_{i // split_step}.csv', 
                        i == split_step)

    # list_timings(MPI.COMM_WORLD, [TimingType.wall])

if __name__ == "__main__":
    main()
