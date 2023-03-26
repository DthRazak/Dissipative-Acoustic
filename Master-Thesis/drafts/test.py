from dolfinx import io
from mpi4py import MPI
from utils.meshing import mesh_to_XDMFFile

import pipe_0_2d
import pipe_0_3d

from segment_builder import SegmentBuilder
from pipe_builder import build_mesh


config = {
    'pipe': {
        'N': 20,
        'type': 1,
        'bubble_radius': 0.05,
        'bubble_lvl': [0.25, 0.1]
    },
    'problem': {
        'fluid': 'Water',
        'contaminant': 'Air',
        'pressure': 1e3,
        'T':  1.7e-2,
        'dt': 8.75e-6,
        'pt': 8.75e-4,
        'delay': 3.5e-3
    },
    'petsc': {
        'solver': 'preonly',
        'pc': 'lu'
    },
    'logs': True,
    'file_prefix': True
}

def main():
    # domain, mt, ft = pipe_0_2d.build_normal_mesh(20, MPI.COMM_WORLD, 0)
    # domain, mt, ft = pipe_0_2d.build_bubble_mesh(20, 0.04, MPI.COMM_WORLD, 0)
    # domain, mt, ft = pipe_0_3d.build_normal_mesh(30, MPI.COMM_WORLD, 0)
    
    domain, mt, ft = build_mesh(config['pipe'], MPI.COMM_WORLD, 0, config['logs'])
    
    mesh_to_XDMFFile(domain, mt, ft, '/root/Meshes/test_mesh.xdmf')
    xdmf_d = io.XDMFFile(domain.comm, '/root/Meshes/test_mesh.xdmf', "w")
    xdmf_d.write_mesh(domain)
    xdmf_d.close()

    # segmentBuilder = SegmentBuilder()

    # centres = segmentBuilder.build(4, 0, 0, 0.05, 0.2)
    # print(centres)


if __name__ == '__main__':
    main()
