import time
from collections import Counter
from operator import methodcaller
from typing import Optional, List, Tuple

from core.MoleculeVisualizer import visualize_molecule
from optimization.genetic.molecule.MoleculeGenerator import MoleculeGenerator
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule


def generate_population(population_size: int, generator: MoleculeGenerator, **kwargs) -> tuple[list[molecule], float]:
    task_start: float = time.perf_counter()
    print('Starting generate_population ...')
    population: list[molecule] = []
    for i in range(population_size):
        population.append(generator.generate_molecule(**kwargs))
    print(f'Missed runs during generating: {generator.missed_runs}')
    task_end: float = time.perf_counter()
    return population, task_end - task_start


def load_initial_guesses(guess_path: str, pop_size: int, input_mol: molecule, cluster_size: int,
                         internuclear_distances) -> List[molecule]:
    print('Checking initial guesses ...')
    population: List[molecule] = []
    if guess_path is not None:
        with open(guess_path, 'r') as f:
            guesses = f.readlines()
        i = 0
        while i < len(guesses):
            if guesses[i].strip().isdigit():
                i, guess = __load_guess(guesses, i)
                if __check_guess_to_input(guess, input_mol, cluster_size):
                    raise RuntimeError(f'Guess #{len(population) + 1} atoms do not match the input cluster atoms')
                if not guess.is_valid(internuclear_distances):
                    valid_min_distance, distances = guess.has_valid_min_distances(internuclear_distances)
                    if not valid_min_distance:
                        raise RuntimeError(f'Initial guess #{len(population) + 1} has invalid minimum'
                                           f' distances:\n{distances}')
                    is_connected, adjacencies = guess.has_valid_connectedness(internuclear_distances)
                    if not is_connected:
                        raise RuntimeError(f'Initial guess #{len(population) + 1} is not connected:\n{adjacencies}')
                population.append(guess)
            else:
                i += 1

    if pop_size < len(population):
        raise RuntimeError(f"Too many guesses provided. Expected at most pop_size {pop_size} but got {len(population)}")
    print(f'Loaded {len(population)} initial guesses')
    return population


def print_population(population: list[molecule], gen: str | int, max_display: Optional[int] = None,
                     sort: bool = True) -> None:
    if max_display is None:
        max_display = len(population)
    pop: list[molecule] = get_best_individuals(population, max_display) if sort else population
    print(f'Generation: {gen}')
    for ind in pop:
        print(ind)


def print_fitness(population: list[molecule], gen: str | int, max_display: Optional[int] = None,
                  sort: bool = True) -> None:
    if max_display is None:
        max_display = len(population)
    pop: list[molecule] = get_best_individuals(population, max_display) if sort else population
    print(f'Generation: {gen}')
    for mol in pop:
        print(f'Fitness: {mol.get_fitness()}')


def get_best_individuals(population: list[molecule], n_ind: int = 1, minimizing: bool = False) -> list[molecule]:
    return sorted(population, key=methodcaller("get_compare_fitness_value"), reverse=minimizing)[:n_ind]


def get_worst_individuals(population: list[molecule], n_ind: int = 1, minimizing: bool = True) -> list[molecule]:
    return sorted(population, key=methodcaller("get_compare_fitness_value"), reverse=minimizing)[:n_ind]


def is_better_fitness(mol1: molecule, mol2: molecule, rmsd_reference: Optional[molecule] = None) -> bool:
    if rmsd_reference is not None:
        return mol1.get_fitness(rmsd_reference) < mol2.get_fitness(rmsd_reference)
    return mol1.get_fitness() < mol2.get_fitness()


def is_better_rmsd_fitness(mol1: molecule, mol2: molecule, rmsd1: float, rmsd2: float) -> bool:
    return mol1.get_fitness(rmsd=rmsd1) < mol2.get_fitness(rmsd=rmsd2)


def set_molecules_for_modification(molecules: list[molecule]) -> list[molecule]:
    new_molecules: list[molecule] = []
    for mol in molecules:
        if mol.is_modified():
            new_molecules.append(mol)
        else:
            new_mol: molecule = mol.copy()
            new_mol.mark_change()
            new_molecules.append(new_mol)
    return new_molecules


def __load_guess(guesses: List[str], i: int) -> Tuple[int, molecule]:
    xyz_length: int = int(guesses[i].strip()) + 1
    xyz_string: str = guesses[i]
    i += 1
    for _ in range(xyz_length):
        xyz_string += guesses[i]
        i += 1
    i += 1
    return i, molecule.from_xyz_string(xyz_string)


def __check_guess_to_input(guess: molecule, input_molecule: molecule, cluster_size: int) -> bool:
    guess_atoms = [atom.element_symbol for atom in guess.atoms]
    input_atoms = [atom.element_symbol for atom in input_molecule.atoms for _ in range(cluster_size)]
    return Counter(guess_atoms) != Counter(input_atoms)


class PopulationStats:
    max_leaderboard: int
    working_directory: str
    leaderboard: List[molecule | None]
    rmsd_threshold: Optional[float]
    rmsd_methods: List[str]

    def __init__(self, working_directory: str, rmsd_methods: List[str], max_leaderboard: int = 5,
                 rmsd_threshold: Optional[float] = None):
        self.max_leaderboard = max_leaderboard
        self.rmsd_threshold = rmsd_threshold
        self.leaderboard = [None] * max_leaderboard
        self.working_directory = working_directory
        self.rmsd_methods = rmsd_methods

    def add_all(self, pop: List[molecule]) -> float:
        task_start: float = time.perf_counter()
        for mol in pop:
            last: Optional[molecule] = self.get_last()
            if last is None or is_better_fitness(mol, last):
                self.__try_new_best(mol)
        return time.perf_counter() - task_start

    def __try_new_best(self, mol: molecule) -> None:
        saved = mol.copy()
        for i in range(self.max_leaderboard):
            current: molecule | None = self.leaderboard[i]
            if current is None:
                self.leaderboard[i] = saved
                print(f'Saving molecule (new energy: {saved.get_fitness()}) to the leaderboard[{i}] (energy: None)')
                return

            is_better: bool = is_better_fitness(saved, current)
            if self.rmsd_threshold is not None:
                rmsd: float = saved.get_rmsd(current, self.rmsd_methods)
                if rmsd <= self.rmsd_threshold:
                    if is_better:
                        print(f'Replacing molecule (new energy: {saved.get_fitness()})'
                              f' at the leaderboard[{i}] (energy: {current.get_fitness()})')
                        self.leaderboard[i] = saved
                    return
            if is_better:
                print(f'Moving molecules one down (new energy: {saved.get_fitness()})'
                      f' at the leaderboard[{i}] (energy: {current.get_fitness()})')
                for j in range(self.max_leaderboard - 1, i, -1):
                    self.leaderboard[j] = self.leaderboard[j - 1]
                self.leaderboard[i] = saved
                return

    def get_best(self) -> molecule | None:
        return self.leaderboard[0]

    def get_last(self) -> molecule | None:
        return self.leaderboard[-1]

    def print_best(self) -> None:
        print(f'Best molecule:\n{self.get_best().get_short_info()}')

    def get_all(self) -> List[molecule]:
        pop: List[molecule] = []
        for i in range(self.max_leaderboard):
            if self.leaderboard[i] is None:
                break
            pop.append(self.leaderboard[i])
        return pop

    def visualize(self):
        for i in range(self.max_leaderboard):
            if self.leaderboard[i] is None:
                break
            print(f'{i + 1}. rank:\n{self.leaderboard[i].get_short_info()}\n')
            visualize_molecule(self.leaderboard[i])
            print('---')

    def save_to_file(self, file_path: str | None = None, file_name: str | None = None) -> None:
        if file_name is None:
            file_name = 'leaderboard'
        if file_path is None:
            file_path = self.working_directory
        with open(f'{file_path}/{file_name}.txt', 'w') as f:
            for i in range(self.max_leaderboard):
                if self.leaderboard[i] is None:
                    break
                self.leaderboard[i].comment = f'{i + 1}. rank, Energy: {self.leaderboard[i].get_fitness():.6f}'
                gradient_norm = self.leaderboard[i].get_gradient_norm()
                if gradient_norm is not None:
                    self.leaderboard[i].comment += f', Gradient norm: {gradient_norm:.6f}'
                f.write(f'{self.leaderboard[i].get_xyz_string()}\n')
        print(f'Saved leaderboard to {file_path}/{file_name}.txt')

    def length(self):
        length: int = 0
        for mol in self.leaderboard:
            if mol is None:
                break
            length += 1
        return length
