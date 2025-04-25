import os
import shutil
import subprocess
import sys
import time
from typing import Optional

from core import MoleculeVisualizer
from core.Checkpoint import RestoredCheckpoint
from core.EvolutionContext import EvolutionContext
from core.MoleculeVisualizer import visualize_two_molecules, visualize_molecule, check_is_ipykernel
from optimization.GeometryOptimizer import GeometryOptimizer
from optimization.genetic.EvolutionEngine import EvolutionEngine
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.PopulationUtils import PopulationStats
from utils.FileUtils import mkdtemp


# Examples of how to run the genetic algorithm:
#   python3 main.py genetic -config data/examples/Au_6/au_6.yaml
#   python3 main.py genetic -config data/examples/4-H_2O/4-h_2o.yaml
def main():
    start = time.perf_counter()

    argc: int = len(sys.argv)
    if argc <= 1:
        print("Missing program command line argument (e.g. 'genetic' or 'optimize' or 'visualize' or 'align')")
        exit(1)

    program: str = sys.argv[1]
    if program.casefold() == 'genetic':
        config_path = None
        checkpoint_path = None
        working_directory = None
        project_directory = None
        for i in range(2, argc):
            if sys.argv[i].casefold() == '-config' and i + 1 < argc:
                config_path = sys.argv[i + 1]
            elif sys.argv[i].casefold() == '-checkpoint' and i + 1 < argc:
                checkpoint_path = sys.argv[i + 1]
            elif sys.argv[i].casefold() == '-work_dir' and i + 1 < argc:
                working_directory = sys.argv[i + 1]
            elif sys.argv[i].casefold() == '-project_dir' and i + 1 < argc:
                project_directory = sys.argv[i + 1]

        if config_path is None:
            print("Missing config command line argument (provide config path 'optimize -config <path>')")
            exit(1)

        genetic_algorithm(config_path, checkpoint_path, working_directory, project_directory)
    elif program.casefold() == 'optimize':
        if argc < 3:
            print("Missing input file command line argument (specify: 'optimize <xyz_path> -threads <threads>"
                  " -steps <max_steps> -program <method_program> -method <method>' -opt_program <optimization_program>)")
            exit(1)
        xyz_path: str = sys.argv[2]
        threads: int = 1
        max_steps: int = 50
        method_program: str = 'orca'
        method: str = 'XTB1'
        optimization_program: str = 'ASE'
        for i in range(3, argc):
            if sys.argv[i].casefold() == '-threads' and i + 1 < argc:
                threads = int(sys.argv[i + 1])
            elif sys.argv[i].casefold() == '-steps' and i + 1 < argc:
                max_steps = int(sys.argv[i + 1])
            elif sys.argv[i].casefold() == '-program' and i + 1 < argc:
                method_program = sys.argv[i + 1]
            elif sys.argv[i].casefold() == '-method' and i + 1 < argc:
                method = sys.argv[i + 1]
            elif sys.argv[i].casefold() == '-opt_program' and i + 1 < argc:
                optimization_program = sys.argv[i + 1]
            else:
                print(f"Unknown or incomplete command line argument: {sys.argv[i]}")
                exit(1)
        optimize(xyz_path, threads, max_steps, method, method_program, optimization_program)
    elif program.casefold() == 'visualize':
        raise NotImplementedError
    elif program.casefold() == 'align':
        if argc < 5:
            print("Missing input files command line argument (specify: 'align <method> <path1> <path2>')")
            exit(1)
        method: str = sys.argv[2]
        path1: str = sys.argv[3]
        path2: str = sys.argv[4]
        align_molecules(path1, path2, method)
    else:
        print("Unknown program command line argument (use 'genetic' or 'optimize' or 'visualize' or 'align')")
        exit(1)

    end = time.perf_counter()
    print(f'\nRun time: {end - start:0.4f} seconds')


def genetic_algorithm(config_path: str, checkpoint_path: Optional[str] = None, working_directory: Optional[str] = None,
                      project_directory: Optional[str] = None):
    context: EvolutionContext = EvolutionContext(config_path, working_directory, project_directory)
    checkpoint: Optional[RestoredCheckpoint] = None
    if checkpoint_path is not None:
        print(f'Restoring state from checkpoint: {checkpoint_path}')
        checkpoint: RestoredCheckpoint = RestoredCheckpoint(checkpoint_path)

    print(f'Using config: {config_path}')
    init_genetic_algorithm(context.config.problem.working_directory)
    engine: EvolutionEngine = context.get_evolution_engine()

    if checkpoint is None:
        stats: PopulationStats = engine.run()
    else:
        stats: PopulationStats = engine.continue_from_checkpoint(checkpoint)
    cleanup(context.config.problem.working_directory)

    print(f"\n************ DISPLAYING RESULTS **************\n")
    stats.save_to_file()
    stats.visualize()

def init_genetic_algorithm(working_directory: str):
    if not os.path.exists(working_directory):
        os.makedirs(working_directory, 0o2777)
        commands = [
            "setfacl -Rdm u::rwx,g::rwx,o::rwx " + working_directory,
            "setfacl -Rm u::rwx,g::rwx,o::rwx " + working_directory
        ]

        for command in commands:
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Successfully executed: {command}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing command: {command}")
                raise e

    if not os.path.exists(f'{working_directory}/temp'):
        mkdtemp(f'{working_directory}/temp')

    debug_dir: str = f'{working_directory}/debug'
    if not os.path.exists(debug_dir):
        mkdtemp(debug_dir)

    MoleculeVisualizer.is_ipykernel = check_is_ipykernel()

def cleanup(working_directory: str):
    shutil.rmtree(f'{working_directory}/temp')

def optimize(path: str, threads: int, max_steps: int, method: str, method_program: str, optimization_program: str):
    print(f'Optimizing molecule "{path}": {method_program} - {method} with {threads} threads, {max_steps} max steps)')
    if not os.path.exists('data/temp'):
        os.makedirs('data/temp')
    go = GeometryOptimizer(method=method, method_program=method_program, optimization_program=optimization_program,
                           working_directory='data')
    mol: molecule = molecule.from_xyz_file(path)
    print(f'Before optimization:\n{mol}\n')
    visualize_molecule(mol)
    mol: molecule = go.optimize_geometry(mol, threads, max_steps, 'temp')
    print(f'After optimization:\n{mol}\n')
    visualize_molecule(mol)

def align_molecules(mol1_path: Optional[str] = None, mol2_path: Optional[str] = None, method: str = 'rdkit'):
    MoleculeVisualizer.is_ipykernel = check_is_ipykernel()
    print('Starting alignment of mol1 with mol2:')
    mol1: molecule = molecule.from_xyz_file(mol1_path)
    mol2: molecule = molecule.from_xyz_file(mol2_path)
    print(f'Mol1:\n{mol1.get_xyz_string()}')
    mol1.view()
    print(f'Mol2:\n{mol2.get_xyz_string()}')
    mol2.view()

    print('Visualizing mol1 and mol2:')
    visualize_two_molecules(mol2.get_xyz_string(), mol1.get_xyz_string())

    aligned_mol: molecule = mol1.align_with(mol2, [method])
    print(f'({method}) Aligned molecule mol1:\n{aligned_mol.get_xyz_string()}')
    print('Visualizing mol2 and aligned mol1:')
    visualize_two_molecules(mol2.get_xyz_string(), aligned_mol.get_xyz_string())
    print(f'RMSD: {aligned_mol.get_rmsd(mol2)}')

if __name__ == '__main__':
    main()
