from random import randint, random, uniform
from typing import List, Optional

import numpy as np
from numpy import ndarray

from core.InternuclearDistances import InternuclearDistances
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.MoleculeUtils import element_coords_to_molecule
from optimization.genetic.molecule.PopulationUtils import set_molecules_for_modification
from optimization.genetic.operations.OperatorUtils import split_centers, unpack_centers, \
    reconnect_group, random_rotation_matrix, rotate_coordinates, reconnect_heteronuclear


class Mutator:
    probability: float
    performed: int = 0
    invalid: int = 0
    infinite: int = 0

    def __init__(self, probability: float):
        self.probability = probability

    def mutate(self, mol: molecule) -> molecule:
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


class CreepMutator(Mutator):
    per_atom_probability: float
    max_strength: float

    def __init__(self, probability: float, per_atom_probability: float = 0.25, max_mutation_strength: float = 1):
        super().__init__(probability)
        self.per_atom_probability = per_atom_probability
        self.max_strength = max_mutation_strength

    def __str__(self) -> str:
        return (f'CreepMutator(probability={self.probability}, '
                f'per_atom_probability={self.per_atom_probability}, '
                f'max_strength={self.max_strength})')

    def mutate(self, mol: molecule) -> molecule:
        if random() < self.probability:
            mutated_mol = self.creep_mutation(mol, self.per_atom_probability, self.max_strength)
            return mutated_mol
        return mol

    @staticmethod
    def creep_mutation(mol: molecule, probability: float, max_strength: float) -> molecule:
        # TODO fix broken structure
        mutated_mol: molecule = set_molecules_for_modification([mol])[0]
        modified: bool = False

        for atom in mutated_mol.atoms:
            if random() < probability:
                coord_index = randint(0, 2)
                mutation_value = uniform(-max_strength, max_strength)
                atom.xyz_coordinates[coord_index] += mutation_value
                modified = True

        return molecule.from_xyz_string(mutated_mol.get_xyz_string()) if modified else mol


class TwinningMutator(Mutator):
    base_mol_size: int
    internuclear_distances: InternuclearDistances
    max_tries: int = 500

    def __init__(self, probability: float, base_mol_size: int, internuclear_distances: InternuclearDistances):
        super().__init__(probability)
        if base_mol_size is None or base_mol_size < 1:
            raise ValueError("Base molecule size must be defined.")
        if internuclear_distances is None:
            raise ValueError("Internuclear distances must be defined.")
        self.base_mol_size = base_mol_size
        self.internuclear_distances = internuclear_distances

    def __str__(self) -> str:
        return f'TwinningMutator(probability={self.probability}, max_tries={self.max_tries})'

    def debug_str(self) -> Optional[str]:
        return (f'TwinningMutator performed {self.performed} mutations. ' +
                f'({self.invalid} invalid mutations, {self.infinite} infinite loops)')

    def mutate(self, mol: molecule) -> molecule:
        if random() < self.probability:
            self.performed += 1
            modified_mol: molecule = set_molecules_for_modification([mol])[0]
            modified_mol = self.twinning_mutation(modified_mol, self.base_mol_size, self.max_tries,
                                                  self.internuclear_distances)
            if modified_mol is None:
                self.infinite += 1
                return mol
            elif not modified_mol.is_valid(self.internuclear_distances):
                self.invalid += 1
                return mol
            return modified_mol
        return mol

    @staticmethod
    def twinning_mutation(mol: molecule, base_size: int, max_tries: int,
                          internuclear_distances: InternuclearDistances) -> Optional[molecule]:
        heteronuclear: bool = mol.is_heteronuclear()
        group1, group2 = split_centers(mol, base_size)
        group1, group2 = unpack_centers(group1, group2)
        element_symbols: List[str] = mol.get_element_symbols().tolist()

        if heteronuclear:
            group1 = reconnect_heteronuclear(group1, element_symbols[:len(group1)], max_tries, internuclear_distances,
                                             base_size)
            group2 = reconnect_heteronuclear(group2, element_symbols[:len(group2)], max_tries, internuclear_distances,
                                             base_size)
        else:
            group1 = reconnect_group(group1, element_symbols[:len(group1)], max_tries, internuclear_distances)
            group2 = reconnect_group(group2, element_symbols[:len(group2)], max_tries, internuclear_distances)
        if group1 is None or group2 is None:
            return None

        rotating_group1: bool = len(group1) <= len(group2)
        rotation_group: ndarray = np.array(group1 if rotating_group1 else group2)
        original_group: ndarray = np.array(group1 if not rotating_group1 else group2)

        rotation_matrix: ndarray = random_rotation_matrix()
        rotated_group: ndarray = rotate_coordinates(rotation_group, rotation_matrix)

        group = original_group.tolist()
        group.extend(rotated_group.tolist())
        if heteronuclear:
            group = reconnect_heteronuclear(group, element_symbols, max_tries, internuclear_distances, base_size)
        else:
            group = reconnect_group(group, element_symbols, max_tries, internuclear_distances)
        if group is None:
            return None

        return element_coords_to_molecule(group, element_symbols)
