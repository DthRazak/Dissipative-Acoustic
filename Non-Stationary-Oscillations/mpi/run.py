#!/usr/bin/env python3

import subprocess


GRID_CONFIG_0 = {
    'script': './2d.py',
    'N': [40],
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': 6.5e-8,
    'fluid': 'Water',
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

# GRID_CONFIG_0 = {
#     'script': './1d.py',
#     'N': [51200],
#     'T': 5.0e-5,
#     't': 1.5e-6,
#     'ct': 2.0e-5,
#     'dt': 6.5e-8,
#     'fluid': 'Water',
#     'solver': ['preonly', 'gmres', 'gcr'],
#     'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
#     'np': [1, 2],
#     'skip_params': []
# }

GRID_CONFIG_1 = {
    'script': './1d.py',
    'N': [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200],
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': 6.5e-8,
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

# TIME_CONFIG_0 = {
#     'script': './1d.py',
#     'N': 800,
#     'T': 5.0e-5,
#     't': 1.5e-6,
#     'ct': 2.0e-5,
#     'dt': [2.0e-7 / 2**i for i in range(5)],
#     'fluid': 'Water',
#     'solver': ['preonly', 'gmres', 'gcr'],
#     'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
#     'np': [1, 2],
#     'skip_params': [
#         # ('gmres', 'bjacobi'),
#         # ('gmres', 'asm'),
#         # ('gcr', 'bjacobi'),
#         # ('gcr', 'asm')
#     ]
# }

TIME_CONFIG_0 = {
    'script': './2d.py',
    'N': 20,
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': [2.0e-7 / 2**i for i in range(5)],
    'fluid': 'Water',
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

TIME_CONFIG_1 = {
    'script': './1d.py',
    'N': 250,
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': [5.0e-7 / 2**i for i in range(8)],
    'fluid': 'Water',
    'solver': ['preonly', 'gmres', 'gcr'],
    'pc': ['cholesky', 'lu', 'bjacobi', 'asm'],
    'np': [1, 2, 4, 8, 16],
    'skip_params': []
}

TIME_CONFIG_2 = {
    'script': './2d.py',
    'N': 20,
    'T': 5.0e-5,
    't': 1.5e-6,
    'ct': 2.0e-5,
    'dt': [5.0e-7 / 2**i for i in range(6)],
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
        file.write('{}\n'.format(';'.join(['solver', 'pc', 'N', 'np', 'L2', 'H1', 'time'])))

        for solver in GRID_CONFIG['solver']:
            for pc in GRID_CONFIG['pc']:
                if (solver, pc) in GRID_CONFIG['skip_params']:
                    continue

                for N in GRID_CONFIG['N']:
                    for np in GRID_CONFIG['np']:
                        command = [
                            '/usr/bin/time', '-f', '%e', 
                            'mpirun', '-np', f'{np}', GRID_CONFIG["script"],
                            '--n', f'{N}', '--T', f'{GRID_CONFIG["T"]}',
                            '--t', f'{GRID_CONFIG["t"]}', '--ct', f'{GRID_CONFIG["ct"]}',
                            '--dt', f'{GRID_CONFIG["dt"]}', '--fluid', GRID_CONFIG['fluid'],
                            '--solver', solver, '--pc', pc, '--no-prefix'
                        ]
                        print(f'Solver: {solver}, PC: {pc}, N: {N}, np: {np}')

                        p = subprocess.run(command, capture_output=True, text=True)
                        
                        print(p.stdout, end='')
                        print(f'Time(s) = {p.stderr}')

                        L2 = str(p.stdout.split('\n')[0].split(' = ')[1])
                        H1 = str(p.stdout.split('\n')[1].split(' = ')[1])
                        time = str(p.stderr[:-1])
                    
                        file.write('{}\n'.format(';'.join([solver, pc, str(N), str(np), L2, H1, time])))

def compute_time_convegence(GRID_CONFIG, filename='time_conv.csv'):
    with open(filename, 'w') as file:
        file.write('{}\n'.format(';'.join(['solver', 'pc', 'dt', 'np', 'L2', 'H1', 'time'])))

        for solver in GRID_CONFIG['solver']:
            for pc in GRID_CONFIG['pc']:
                if (solver, pc) in GRID_CONFIG['skip_params']:
                    continue

                for dt in GRID_CONFIG['dt']:
                    for np in GRID_CONFIG['np']:
                        command = [
                            '/usr/bin/time', '-f', '%e', 
                            'mpirun', '-np', f'{np}', GRID_CONFIG["script"],
                            '--n', f'{GRID_CONFIG["N"]}', '--T', f'{GRID_CONFIG["T"]}',
                            '--t', f'{GRID_CONFIG["t"]}', '--ct', f'{GRID_CONFIG["ct"]}',
                            '--dt', f'{dt}', '--fluid', GRID_CONFIG['fluid'],
                            '--solver', solver, '--pc', pc, '--no-prefix'
                        ]
                        print(f'Solver: {solver}, PC: {pc}, dt: {dt}, np: {np}')

                        p = subprocess.run(command, capture_output=True, text=True)
                        
                        print(p.stdout, end='')
                        print(f'Time(s) = {p.stderr}')

                        L2 = str(p.stdout.split('\n')[0].split(' = ')[1])
                        H1 = str(p.stdout.split('\n')[1].split(' = ')[1])
                        time = str(p.stderr[:-1])
                    
                        file.write('{}\n'.format(';'.join([solver, pc, str(dt), str(np), L2, H1, time])))


def main():
    compute_grid_convegence(GRID_CONFIG_1, './results/grid_conv_1d.csv')
    compute_grid_convegence(GRID_CONFIG_2, './results/grid_conv_2d.csv')
    compute_time_convegence(TIME_CONFIG_1, './results/time_conv_1d.csv')
    compute_time_convegence(TIME_CONFIG_2, './results/time_conv_2d.csv')


if __name__ == "__main__":
    main()
