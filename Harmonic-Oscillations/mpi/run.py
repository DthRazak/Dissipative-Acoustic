#!/usr/bin/env python3

import subprocess


GRID_CONFIG_0 = {
    'script': './2d.py',
    'N': [80],
    'freq': 1.0e6,
    'fluid': 'Fuel oil',
    'solver': ['preonly', 'gmres', 'gcr'],
    'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
    'np': [1, 2],
    'skip_params': [
        ('gmres', 'bjacobi'),
        ('gmres', 'asm'),
        ('gcr', 'bjacobi'),
        ('gcr', 'asm')
    ]
}

GRID_CONFIG_1 = {
    'script': './1d.py',
    'N': [100 * 2**i for i in range(10)],
    'freq': 5.0e5,
    'fluid': 'Water',
    'solver': ['preonly', 'gmres', 'gcr'],
    'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
    'np': [1, 2, 4, 8, 16],
    'skip_params': []
}

GRID_CONFIG_2 = {
    'script': './2d.py',
    'N': [20, 40, 80],
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': 6.5e-7,
    'fluid': 'Water',
    'solver': ['preonly', 'gmres', 'gcr'],
    'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
    'np': [1, 2, 4, 8, 16],
    'skip_params': [
        ('gmres', 'bjacobi'),
        ('gmres', 'asm'),
        ('gcr', 'bjacobi'),
        ('gcr', 'asm')
    ]
}

def compute_grid_convegence(GRID_CONFIG, filename='grig_conv.csv'):
    with open(filename, 'w') as file:
        file.write('{}\n'.format(';'.join([
            'solver', 'pc', 'N', 'np', 'L2_r', 'H1_r', 'L2_c', 'H1_c', 'time'
        ])))

        for solver in GRID_CONFIG['solver']:
            for pc in GRID_CONFIG['pc']:
                if (solver, pc) in GRID_CONFIG['skip_params']:
                    continue

                for N in GRID_CONFIG['N']:
                    for np in GRID_CONFIG['np']:
                        command = [
                            '/usr/bin/time', '-f', '%e', 
                            'mpirun', '-np', f'{np}', GRID_CONFIG["script"],
                            '--n', f'{N}', '--freq', f'{GRID_CONFIG["freq"]}',
                            '--fluid', GRID_CONFIG['fluid'],
                            '--solver', solver, '--pc', pc, '--no-prefix'
                        ]
                        print(f'Solver: {solver}, PC: {pc}, N: {N}, np: {np}')

                        p = subprocess.run(command, capture_output=True, text=True)
                        
                        print(p.stdout, end='')
                        print(f'Time(s) = {p.stderr}')

                        L2_r = str(p.stdout.split('\n')[0].split(' = ')[1])
                        H1_r = str(p.stdout.split('\n')[1].split(' = ')[1])
                        L2_c = str(p.stdout.split('\n')[2].split(' = ')[1])
                        H1_c = str(p.stdout.split('\n')[3].split(' = ')[1])
                        time = str(p.stderr[:-1])
                    
                        file.write('{}\n'.format(';'.join([
                            solver, pc, str(N), str(np), L2_r, H1_r, L2_c, H1_c, time
                        ])))

def main():
    # compute_grid_convegence(GRID_CONFIG_0, './results/grid_conv_2d.csv')
    compute_grid_convegence(GRID_CONFIG_1, './results/grid_conv_1d.csv')
    compute_grid_convegence(GRID_CONFIG_2, './results/grid_conv_2d.csv')


if __name__ == "__main__":
    main()
