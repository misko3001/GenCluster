from random import random, randint, uniform
from typing import List, Tuple, Set, Optional

import networkx as nx
import numpy as np
from mlatom import atom
from numpy import ndarray

from core.InternuclearDistances import InternuclearDistances
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule, MoleculeIndividual
from optimization.genetic.molecule.MoleculeUtils import get_distances_and_adjacency_matrix, check_min_distances, \
    element_coords_to_molecule
from optimization.genetic.molecule.PopulationUtils import set_molecules_for_modification
from optimization.genetic.operations.OperatorUtils import split_centers_and_pair, unpack_centers, \
    reconnect_group, reconnect_heteronuclear


class Crossover:
    probability: float
    performed: int = 0
    invalid: int = 0
    infinite: int = 0

    def __init__(self, probability: float):
        self.probability = probability

    def crossover(self, mol1: molecule, mol2: molecule) -> tuple[molecule, molecule]:
        raise NotImplementedError

    def debug_str(self) -> Optional[str]:
        return None

    def get_stats(self) -> List[int]:
        return [self.performed, self.invalid, self.infinite]

    def update_stats(self, debug_values: List[int]):
        self.performed = debug_values[0]
        self.invalid = debug_values[1]
        self.infinite = debug_values[2]

    def get_name(self) -> str:
        return self.__class__.__name__


class TwoPointCrossover(Crossover):
    internuclear_distances: InternuclearDistances
    alignment_methods: List[str]

    def __init__(self, probability: float, internuclear_distances: InternuclearDistances, alignment_methods: List[str]):
        super().__init__(probability)
        if internuclear_distances is None:
            raise RuntimeError('Missing internuclear_distances for TwoPointCrossover')
        self.internuclear_distances = internuclear_distances
        self.alignment_methods = alignment_methods

    def __str__(self) -> str:
        return f'TwoPointCrossover(probability={self.probability})'

    def crossover(self, mol1: molecule, mol2: molecule) -> Tuple[molecule, molecule]:
        if random() < self.probability:
            modified_molecules = set_molecules_for_modification([mol1, mol2])
            ch1, ch2 = self.two_point_crossover(modified_molecules[0], modified_molecules[1], self.alignment_methods,
                                                self.internuclear_distances)
            if not ch1.is_valid(self.internuclear_distances):
                self.invalid += 1
                ch1 = mol1
            if not ch2.is_valid(self.internuclear_distances):
                self.invalid += 1
                ch2 = mol2
            return ch1, ch2
        return mol1, mol2

    @staticmethod
    def two_point_crossover(mol1: molecule, mol2: molecule, alignment_methods: List[str],
                            internuclear_distances: InternuclearDistances) -> Tuple[molecule, molecule]:
        # TODO remake for molecules
        child1 = mol1
        child2 = mol2.align_with(child1, alignment_methods)

        atoms1: list[atom] = child1.atoms.copy()
        atoms2: list[atom] = child2.atoms.copy()

        if len(atoms1) != len(atoms2):
            raise ValueError("Molecules must have the same amount of atoms.")

        point1: int = randint(1, len(atoms1) - 2)
        point2: int = randint(point1 + 1, len(atoms2) - 1)

        child1.atoms = atoms1[:point1] + atoms2[point1:point2] + atoms1[point2:]
        child2.atoms = atoms2[:point1] + atoms1[point1:point2] + atoms2[point2:]

        element_symbols: List[str] = [at.element_symbol for at in child1.atoms]
        child1 = TwoPointCrossover.reconnect(child1.get_xyz_coordinates(), element_symbols, internuclear_distances)
        child2 = TwoPointCrossover.reconnect(child2.get_xyz_coordinates(), element_symbols, internuclear_distances)
        return child1, child2

    @staticmethod
    def reconnect_atoms(coords: ndarray, element_symbols: List[str], internuclear_distances: InternuclearDistances,
                        distances: ndarray, adjacency_matrix: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        adjacency_graph = nx.from_numpy_array(adjacency_matrix)
        components: List[Set[int]] = list(nx.connected_components(adjacency_graph))

        while not nx.is_connected(adjacency_graph):
            base_component: Set[int] = components[0]
            for component in components:
                if len(component) > len(base_component):
                    base_component = component
            components.remove(base_component)

            disconnected_component = components[0]
            closest_pair: Optional[Tuple[int, int]] = None
            min_distance = float('inf')
            for base_atom_index in base_component:
                for disconnected_atom_index in disconnected_component:
                    atom_distance = distances[base_atom_index][disconnected_atom_index]
                    can_bond = internuclear_distances.get(element_symbols[base_atom_index],
                                                          element_symbols[disconnected_atom_index]) is not None
                    if can_bond and atom_distance < min_distance:
                        min_distance = atom_distance
                        closest_pair = (base_atom_index, disconnected_atom_index)

            if closest_pair is None:
                raise RuntimeError(f"Could not form a bond between two components {base_component} and {disconnected_component}.")

            base_symbol: str = element_symbols[closest_pair[0]]
            disconnected_symbol: str = element_symbols[closest_pair[1]]
            allowed_distance: Tuple[float, float] = internuclear_distances.get(base_symbol, disconnected_symbol)

            current_distance = distances[closest_pair[0]][closest_pair[1]]
            target_distance = uniform(allowed_distance[0], allowed_distance[1])
            distance_to_move = current_distance - target_distance

            direction = coords[closest_pair[0]] - coords[closest_pair[1]]
            translation_vector = (direction / np.linalg.norm(direction)) * distance_to_move

            coords[closest_pair[1]] += translation_vector

            distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                             internuclear_distances)
            adjacency_graph = nx.from_numpy_array(adjacency_matrix)
            components: List[Set[int]] = list(nx.connected_components(adjacency_graph))

        return coords, distances, adjacency_matrix

    @staticmethod
    def reconnect(coords: ndarray, element_symbols: List[str],
                  internuclear_distances: InternuclearDistances) -> molecule:
        distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                         internuclear_distances)
        coords, distances, adjacency_matrix = TwoPointCrossover.reconnect_atoms(coords, element_symbols,
                                                                                internuclear_distances, distances,
                                                                                adjacency_matrix)

        min_constraint = check_min_distances(distances, element_symbols, internuclear_distances)
        while not min_constraint[0]:
            atom1_index: int = min_constraint[1]
            atom2_index: int = min_constraint[2]

            symbol1: str = element_symbols[atom1_index]
            symbol2: str = element_symbols[atom2_index]
            allowed_distance: Tuple[float, float] = internuclear_distances.get(symbol1, symbol2)

            current_distance = distances[atom1_index][atom2_index]
            target_distance = uniform(allowed_distance[0], allowed_distance[1])
            distance_to_move = abs(target_distance - current_distance)

            direction = coords[atom2_index] - coords[atom1_index]
            direction_norm = np.linalg.norm(direction) + 1e-11
            translation_vector = (direction / direction_norm) * distance_to_move

            coords[atom2_index] += translation_vector

            distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                             internuclear_distances)
            coords, distances, adjacency_matrix = TwoPointCrossover.reconnect_atoms(coords, element_symbols,
                                                                                    internuclear_distances, distances,
                                                                                    adjacency_matrix)
            min_constraint = check_min_distances(distances, element_symbols, internuclear_distances)

        return TwoPointCrossover.array_to_molecule(coords, element_symbols)

    @staticmethod
    def array_to_molecule(coords: ndarray, symbols: List[str]) -> molecule:
        xyz_string: str = f'{len(symbols)}\n\n'
        for i in range(len(symbols)):
            xyz_string += f'{symbols[i]} {coords[i][0]} {coords[i][1]} {coords[i][2]}\n'
        return MoleculeIndividual.from_xyz_string(xyz_string)


class SpliceCrossover(Crossover):
    internuclear_distances: InternuclearDistances
    base_mol_size: int
    max_tries: int = 500

    def __init__(self, probability: float, base_mol_size: int, internuclear_distances: InternuclearDistances):
        super().__init__(probability)
        if base_mol_size is None or base_mol_size < 1:
            raise ValueError("Base molecule size must be defined.")
        if internuclear_distances is None:
            raise RuntimeError('Missing internuclear_distances for SpliceCrossover')
        self.internuclear_distances = internuclear_distances
        self.base_mol_size = base_mol_size

    def __str__(self) -> str:
        return f'SpliceCrossover(probability={self.probability}, max_tries={self.max_tries})'

    def debug_str(self) -> Optional[str]:
        return (f'SpliceCrossover performed {self.performed} crossovers. ' +
                f'({self.invalid} invalid children, {self.infinite} infinite loops)')

    def crossover(self, mol1: molecule, mol2: molecule) -> Tuple[molecule, molecule]:
        if random() < self.probability:
            self.performed += 1
            modified_molecules = set_molecules_for_modification([mol1, mol2])
            ch1, ch2 = self.splice_crossover(modified_molecules[0], modified_molecules[1], self.base_mol_size,
                                             self.max_tries, self.internuclear_distances)

            if ch1 is None:
                self.infinite += 1
                ch1 = mol1
            elif not mol1.is_valid(self.internuclear_distances):
                self.invalid += 1
                ch1 = mol1

            if ch2 is None:
                self.infinite += 1
                ch2 = mol2
            elif not mol2.is_valid(self.internuclear_distances):
                self.invalid += 1
                ch2 = mol2

            return ch1, ch2
        return mol1, mol2

    @staticmethod
    def splice_crossover(mol1: molecule, mol2: molecule, mol_size: int, max_tries: int,
                         internuclear_distances: InternuclearDistances) -> Tuple[Optional[molecule], Optional[molecule]]:
        heteronuclear: bool = mol1.is_heteronuclear()
        # Using rdkit so heteronuclear cluster atoms are not rearranged
        aligned_mol2: molecule = mol2.align_with(mol1, ['rdkit'])
        ch1_group, ch2_group = split_centers_and_pair(mol1, aligned_mol2, mol_size)

        ch1_group, ch2_group = unpack_centers(ch1_group, ch2_group)
        element_symbols_1: List[str] = mol1.get_element_symbols().tolist()
        element_symbols_2: List[str] = mol2.get_element_symbols().tolist()

        if heteronuclear:
            ch1_group = reconnect_heteronuclear(ch1_group, element_symbols_1, max_tries, internuclear_distances,
                                                mol_size)
        else:
            ch1_group = reconnect_group(ch1_group, element_symbols_1, max_tries, internuclear_distances)
        ch1: Optional[molecule] = None
        if ch1_group is not None:
            ch1 = element_coords_to_molecule(ch1_group, element_symbols_1)

        if heteronuclear:
            ch2_group = reconnect_heteronuclear(ch2_group, element_symbols_2, max_tries, internuclear_distances,
                                                mol_size)
        else:
            ch2_group = reconnect_group(ch2_group, element_symbols_2, max_tries, internuclear_distances)
        ch2: Optional[molecule] = None
        if ch2_group is not None:
            ch2 = element_coords_to_molecule(ch2_group, element_symbols_2)

        return ch1, ch2
