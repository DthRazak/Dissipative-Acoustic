#!/usr/bin/env python3

from distutils.cmd import Command
import subprocess


GRID_CONFIG_1 = {
    'N': [100, 200, 400, 800, 1600, 3200],#, 6400, 12800],
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': 6.5e-8,
    'fluid': 'Water',
    'solver': ['PREONLY', 'RICHARDSON', 'CG', 'GMRES', 'TFQMR', 'CR'],
    'pc': ['ASM', 'BJACOBI', 'GASM', 'GAMG'],
    'np': [1, 2, 4]
}

GRID_CONFIG_2 = {
    'N': [800],
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': 6.5e-9,
    'fluid': 'Water',
    'solver': ['CG',],
    'pc': ['ASM', 'BJACOBI', 'GASM'],
    'np': [1, 2, 4]
}

def compute_grid_convegence(GRID_CONFIG, filename='grig_conv.csv'):
    with open(filename, 'w') as file:
        file.writelines([';'.join(['solver', 'pc', 'N', 'np', 'L2', 'H1', 'time']), '\n'])

        for solver in GRID_CONFIG['solver']:
            for pc in GRID_CONFIG['pc']:
                for N in GRID_CONFIG['N']:
                    for np in GRID_CONFIG['np']:
                        command = [
                            '/usr/bin/time', '-f', '%e', 
                            'mpirun', '-np', f'{np}', './1d.py',
                            '--n', f'{N}', '--T', f'{GRID_CONFIG["T"]}',
                            '--t', f'{GRID_CONFIG["t"]}', '--ct', f'{GRID_CONFIG["ct"]}',
                            '--dt', f'{GRID_CONFIG["dt"]}', '--fluid', GRID_CONFIG['fluid'],
                            '--solver', solver, '--pc', pc
                        ]
                        print(f'Solver: {solver}, PC: {pc}, N: {N}, np: {np}')

                        p = subprocess.run(command, capture_output=True, text=True)
                        
                        print(p.stdout, end='')
                        print(f'Time(s) = {p.stderr}')

                        L2 = str(p.stdout.split('\n')[0].split(' = ')[1])
                        H1 = str(p.stdout.split('\n')[1].split(' = ')[1])
                        time = str(p.stderr[:-1])

                        file.writelines([';'.join([solver, pc, str(N), str(np), L2, H1, time]), '\n'])


def main():
    compute_grid_convegence(GRID_CONFIG_2)


if __name__ == "__main__":
    main()
