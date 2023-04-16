#!/usr/bin/env python3

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Domain name", default='frequency')
    parser.add_argument('-r', nargs=2, type=int, help="Input file range",
                        default=[0, 1000], metavar=('a', 'b'))

    args = parser.parse_args()

    if args.name == 'frequency':
        for i in range(args.r[0], args.r[1]):
            command = ['mpirun', '-np', '2', './RectangleModel-FrequencyDomain-MPI.py',
                       '--i', f'{i}']
            p = subprocess.run(command, capture_output=True, text=True)
                        
            print(p.stdout, end='')
    elif args.name == 'time':
        for i in range(args.r[0], args.r[1]):
            command = ['./RectangleModel-TimeDomain-MPI.py',
                       '--i', f'{i}']
            p = subprocess.run(command, capture_output=True, text=True)
                        
            print(p.stdout, end='')

if __name__ == "__main__":
    main()
