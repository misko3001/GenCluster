import ast
import re
from typing import List, Optional, Dict, Tuple

import numpy as np

from optimization.GeometryOptimizer import GeometryOptimizer
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.PopulationUtils import PopulationStats
from optimization.genetic.operations.Crossover import Crossover
from optimization.genetic.operations.Mutator import Mutator
from optimization.genetic.operations.Terminator import Terminator

checkpoint_name: str = 'checkpoint.txt'


class RestoredCheckpoint:
    generation: int
    operator_stats: Dict[str, List[any]]
    terminator_stats: Dict[str, List[any]]
    time_stats: any
    evaluator_stats: List[int]
    optimizer_stats: Optional[List[int]]
    current_pop: List[molecule]
    leaderboard: List[molecule]

    def __init__(self, checkpoint_path: str):
        with open(checkpoint_path, 'r') as file:
            lines = file.readlines()

        i: int = 0
        while i < len(lines):
            line: str = lines[i].strip()
            if line == '':
                pass
            elif line.startswith('Generation:'):
                self.generation = int(line.split(' ')[1])
            elif line.startswith('*** OPERATOR STATS ***'):
                i = self.__parse_operator_stats(lines, i)
            elif line.startswith('*** TERMINATOR STATS ***'):
                i = self.__parse_terminator_stats(lines, i)
            elif line.startswith('*** TIME STATS ***'):
                i = self.__parse_time_stats(lines, i)
            elif line.startswith('*** EVALUATOR STATS ***'):
                i = self.__parse_evaluator_stats(lines, i)
            elif line.startswith('*** OPTIMIZER STATS ***'):
                i = self.__parse_optimizer_stats(lines, i)
            elif line.startswith('*** CURRENT POPULATION ***'):
                i = self.__parse_population(lines, i)
            elif line.startswith('*** LEADERBOARD STATS ***'):
                i = self.__parse_leaderboard(lines, i)
            i += 1

    def __parse_leaderboard(self, lines: List[str], i: int) -> int:
        i += 1
        line: str = lines[i].strip()
        leaderboard_size: int = int(line)
        leaderboard: List[molecule] = []
        for _ in range(leaderboard_size):
            i, mol = self.__parse_molecule(lines, i)
            leaderboard.append(mol)
        self.leaderboard = leaderboard
        return i

    def __parse_population(self, lines: List[str], i: int) -> int:
        i += 1
        line: str = lines[i].strip()
        pop: List[molecule] = []
        pop_size: int = int(line)
        i += 1
        for _ in range(pop_size):
            i, mol = self.__parse_molecule(lines, i)
            pop.append(mol)
        self.current_pop = pop
        return i

    @staticmethod
    def __parse_molecule(lines: List[str], i: int) -> Tuple[int, molecule]:
        line: str = lines[i].strip()
        while not line.startswith('XYZ COORDS'):
            i += 1
            line: str = lines[i].strip()
        i, mol = RestoredCheckpoint.__parse_mol_from_xyz_string(lines, i)

        line = lines[i].strip()
        while not line.startswith('AGE:'):
            if not line:
                pass
            elif line.startswith('ENERGY GRADIENTS'):
                i, gradients = RestoredCheckpoint.__parse_energy_gradients(lines, i)
                mol.energy_gradients = gradients
            elif line.startswith('ENERGY:'):
                energy: float = float(line.split(' ')[1])
                mol.energy = energy
            i += 1
            line: str = lines[i].strip()
        age: int = int(line.split(' ')[1])
        mol.age = age
        i += 1
        return i, mol

    @staticmethod
    def __parse_energy_gradients(lines: List[str], i: int) -> Tuple[int, np.ndarray]:
        i += 1
        matrix_string: str = lines[i] + ','
        i += 1
        while not lines[i].strip().endswith(']]'):
            matrix_string += lines[i].strip() + ','
            i += 1
        matrix_string += lines[i]
        i += 1
        matrix_list = ast.literal_eval(re.sub(r'(\d)\s+(-?\d)', r'\1, \2', matrix_string))
        return i, np.array(matrix_list)

    @staticmethod
    def __parse_mol_from_xyz_string(lines: List[str], i: int) -> Tuple[int, molecule]:
        i += 1
        xyz_length: int = int(lines[i].strip()) + 1
        xyz_string: str = lines[i]
        i += 1
        for _ in range(xyz_length):
            xyz_string += lines[i]
            i += 1
        i += 1
        return i, molecule.from_xyz_string(xyz_string)

    def __parse_optimizer_stats(self, lines: List[str], i: int) -> int:
        i, stats = self.__parse_geometry_stats(lines, i)
        self.optimizer_stats = stats
        return i

    def __parse_evaluator_stats(self, lines: List[str], i: int) -> int:
        i, stats = self.__parse_geometry_stats(lines, i)
        self.evaluator_stats = stats
        return i

    @staticmethod
    def __parse_geometry_stats(lines: List[str], i: int) -> Tuple[int, List[int]]:
        i += 1
        line: str = lines[i].strip()
        operations: Optional[int] = None
        failed: Optional[int] = None
        if line.startswith('Operations:'):
            operations = int(line.split(' ')[1])
        elif line.startswith('Failed:'):
            failed = int(line.split(' ')[1])
        i += 1
        line = lines[i].strip()
        if line.startswith('Operations:'):
            operations = int(line.split(' ')[1])
        elif line.startswith('Failed:'):
            failed = int(line.split(' ')[1])
        i += 1
        return i, [operations, failed]

    def __parse_time_stats(self, lines: List[str], i: int) -> int:
        from optimization.genetic.EvolutionEngine import TimeStats
        i += 1
        line: str = lines[i].strip()
        time_stats: TimeStats = TimeStats()
        while line:
            name, time = line.split(' ', 1)
            time_stats.log_time(name, float(time))
            i += 1
            line: str = lines[i].strip()
        self.time_stats = time_stats
        return i

    def __parse_terminator_stats(self, lines: List[str], i: int) -> int:
        i, terminator_stats = self.__parse_stats(lines, i)
        self.terminator_stats = terminator_stats
        return i

    def __parse_operator_stats(self, lines: List[str], i: int) -> int:
        i, operator_stats = self.__parse_stats(lines, i)
        self.operator_stats = operator_stats
        return i

    @staticmethod
    def __parse_stats(lines: List[str], i: int) -> Tuple[int, Dict[str, List[int]]]:
        i += 1
        line: str = lines[i].strip()
        stats: Dict[str, List[any]] = {}
        while line:
            name, list_values = line.split(' ', 1)
            stats[name] = ast.literal_eval(list_values)
            i += 1
            line: str = lines[i].strip()
        return i, stats


def save_checkpoint(working_directory: str, current_pop: List[molecule], pop_stats: PopulationStats,
                    time_stats: any, generation: int, crossover: Crossover, terminators: List[Terminator],
                    mutators: List[Mutator], evaluator: GeometryOptimizer, optimizer: Optional[GeometryOptimizer]):
    checkpoint_path: str = f'{working_directory}/{checkpoint_name}'

    with open(checkpoint_path, 'w') as f:
        f.write(f'Generation: {generation}\n\n')

        f.write('*** OPERATOR STATS ***\n')
        f.write(f'{crossover.get_name()} {crossover.get_stats()}\n')
        for mutator in mutators:
            f.write(f'{mutator.get_name()} {mutator.get_stats()}\n')

        f.write('\n*** TERMINATOR STATS ***\n')
        for terminator in terminators:
            if terminator.get_terminator_values() is not None:
                f.write(f'{terminator.get_name()} {terminator.get_terminator_values()}\n')

        f.write('\n*** TIME STATS ***\n')
        f.write(f'{time_stats.__str__()}')

        f.write('\n*** EVALUATOR STATS ***\n')
        f.write(f'Operations: {evaluator.evals}\nFailed: {evaluator.failed_evals}\n')

        if optimizer is not None:
            f.write('\n*** OPTIMIZER STATS ***\n')
            f.write(f'Operations: {optimizer.optimizations}\nFailed: {optimizer.failed_optimizations}\n')

        f.write('\n*** CURRENT POPULATION ***\n')
        f.write(f'{len(current_pop)}\n')
        for mol in current_pop:
            f.write(f'XYZ COORDS\n')
            f.write(f'{mol.get_xyz_string()}\n')
            gradients = mol.get_energy_gradients()
            if not np.isnan(gradients[0][0]):
                f.write(f'ENERGY GRADIENTS\n')
                f.write(f'{gradients}\n\n')
            f.write(f'ENERGY: {mol.get_fitness()}\n')
            f.write(f'AGE: {mol.age}\n\n')

        f.write('\n*** LEADERBOARD STATS ***\n')
        f.write(f'{pop_stats.length()}\n')
        for mol in pop_stats.get_all():
            f.write(f'XYZ COORDS\n')
            f.write(f'{mol.get_xyz_string()}\n')
            gradients = mol.get_energy_gradients()
            if not np.isnan(gradients[0][0]):
                f.write(f'ENERGY GRADIENTS\n')
                f.write(f'{gradients}\n\n')
            f.write(f'ENERGY: {mol.get_fitness()}\n')
            f.write(f'AGE: {mol.age}\n\n')
