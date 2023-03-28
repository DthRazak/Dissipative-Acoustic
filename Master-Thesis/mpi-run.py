#!/usr/bin/env python3

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Domain name", default='frequency')

    args = parser.parse_args()

    if args.name == 'frequency':
        for i in range(1000):
            command = ['mpirun', '-np', '2', './RectangleModel-FrequencyDomain-MPI.py',
                       '--i', f'{i}']
            p = subprocess.run(command, capture_output=True, text=True)
                        
            print(p.stdout, end='')
    elif args.name == 'time':
        pass

if __name__ == "__main__":
    main()
