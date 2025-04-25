from typing import Tuple, List

import numpy as np
from mlatom import atom
from numpy import ndarray, bool_

from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.MoleculeUtils import element_coords_to_molecule, get_molecule_center_of_mass
from optimization.genetic.operations.OperatorUtils import reconnect_group, reconnect_heteronuclear, \
    random_rotation_matrix, rotate_coordinates


class MoleculeGenerator:
    com_angstrom_distances: Tuple[float, float]
    internuclear_distances: any
    missed_runs: int = 0
    max_invalid_generations: int = 250
    max_missed_runs_per_mol: int = 20
    max_fix_tries: int = 1000

    def __init__(self, com_angstrom_distances: Tuple[float, float], internuclear_distances: any):
        self.com_angstrom_distances = com_angstrom_distances
        self.internuclear_distances = internuclear_distances

    def generate_molecule(self, **kwargs):
        raise NotImplementedError


class PackMolGenerator(MoleculeGenerator):
    instructions: [str] = ['inside box', 'inside cube', 'inside sphere']

    def __str__(self) -> str:
        return f'PackMolGenerator(instructions={self.instructions})'

    def generate_molecule(self, input_molecule: molecule, cluster_size: int):
        raise NotImplementedError


AtomCoords = Tuple[str, float, float, float]
MoleculeStructure = List[AtomCoords]
CentroidArray = List[ndarray]


class RandomGeometryGenerator(MoleculeGenerator):

    def __str__(self) -> str:
        return f'RandomGeometryGenerator(max_missed_runs_per_mol={self.max_missed_runs_per_mol})'

    def generate_molecule(self, input_molecule: molecule, cluster_size: int,
                          internuclear_distances: any):
        missed: int = 0
        heteronuclear: bool = input_molecule.is_heteronuclear()
        while True:
            structure: MoleculeStructure = self.generate_random_structure(input_molecule, cluster_size, heteronuclear)
            mol: molecule = self.xyz_structure_to_molecule(structure)
            mol = self.__fix(mol, input_molecule.get_number_of_atoms(), heteronuclear)
            if mol.is_valid(internuclear_distances):
                mol.charge = input_molecule.charge
                mol.multiplicity = input_molecule.multiplicity
                mol.comment = ''
                return mol
            else:
                missed += 1
                self.missed_runs += 1
                if missed > self.max_missed_runs_per_mol:
                    raise RuntimeError(f'Unable to generate valid molecule within '
                                       f'{self.max_missed_runs_per_mol} iterations.')

    def generate_random_structure(self, mol: molecule, cluster_size: int, heteronuclear: bool) -> MoleculeStructure:
        mol_structure: MoleculeStructure = []
        centroids: CentroidArray = []
        self.add_atoms_to_structure(mol_structure, mol.atoms)
        centroids.append(get_molecule_center_of_mass(mol.get_xyz_coordinates(), mol.get_nuclear_masses()))
        for i in range(1, cluster_size):
            generated_molecule: MoleculeStructure = self.generate_new_molecule(mol, centroids, heteronuclear)
            self.merge_structures(mol_structure, generated_molecule)
        return mol_structure

    @staticmethod
    def add_atoms_to_structure(structure: MoleculeStructure, atoms: List[atom]) -> None:
        for mol_atom in atoms:
            coords: List[float] = mol_atom.xyz_coordinates
            atom_coords: AtomCoords = mol_atom.element_symbol, coords[0], coords[1], coords[2]
            structure.append(atom_coords)

    @staticmethod
    def merge_structures(merge_structure: MoleculeStructure, structure: MoleculeStructure) -> MoleculeStructure:
        for atom_coords in structure:
            merge_structure.append(atom_coords)
        return merge_structure

    @staticmethod
    def check_distances(centroids: List[ndarray], candidate_centroid: ndarray,
                        angstrom_distances: Tuple[float, float]) -> bool_:
        distances = [np.linalg.norm(candidate_centroid - centroid) for centroid in centroids]
        return (np.all(d >= angstrom_distances[0] for d in distances) and
                np.any(d <= angstrom_distances[1] for d in distances))

    def generate_new_molecule(self, input_molecule: molecule, centroids: CentroidArray,
                              heteronuclear: bool) -> MoleculeStructure:
        min_angstrom, max_angstrom = self.com_angstrom_distances
        generated_molecule: molecule = input_molecule.copy()
        if heteronuclear:
            rotated_coords = rotate_coordinates(generated_molecule.get_xyz_coordinates(), random_rotation_matrix()).tolist()
            generated_molecule = element_coords_to_molecule(rotated_coords,
                                                            generated_molecule.get_element_symbols().tolist())
        last_centroid: ndarray = centroids[-1]

        invalid_centroid: int = 0
        while True:
            # Random bond length within the specified range
            bond_length = np.random.uniform(min_angstrom, max_angstrom)

            # Random angles (theta: [0, 2π], phi: [0, π])
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            # Spherical to Cartesian conversion
            x = bond_length * np.sin(phi) * np.cos(theta)
            y = bond_length * np.sin(phi) * np.sin(theta)
            z = bond_length * np.cos(phi)
            new_centroid = np.array([last_centroid[0] + x, last_centroid[1] + y, last_centroid[2] + z])

            if self.check_distances(centroids, new_centroid, self.com_angstrom_distances):
                break
            elif invalid_centroid > self.max_invalid_generations:
                raise RuntimeError(f'Unable to generate centroid within {self.max_invalid_generations} iterations.')
            else:
                invalid_centroid += 1

        # Calculate translation vector
        translation_vector = new_centroid - centroids[0]

        # Translate all atoms in the input molecule
        for generated_atom in generated_molecule.atoms:
            generated_atom.xyz_coordinates += translation_vector

        generated_structure: MoleculeStructure = []
        self.add_atoms_to_structure(generated_structure, generated_molecule.atoms)
        centroids.append(new_centroid)
        return generated_structure

    def __fix(self, mol: molecule, mol_size: int, heteronuclear: bool) -> molecule:
        coords = mol.get_xyz_coordinates().tolist()
        element_symbols: List[str] = mol.get_element_symbols().tolist()
        if heteronuclear:
            reconnected = reconnect_heteronuclear(coords, element_symbols, self.max_fix_tries,
                                                  self.internuclear_distances, mol_size)
        else:
            reconnected = reconnect_group(coords, element_symbols, self.max_fix_tries, self.internuclear_distances)
        if reconnected is None:
            return mol
        return element_coords_to_molecule(reconnected, element_symbols)

    @staticmethod
    def xyz_structure_to_molecule(structure: MoleculeStructure) -> molecule:
        xyz_string: str = f'{len(structure)}\n\n'
        for atom_coords in structure:
            xyz_string += f'{atom_coords[0]} {atom_coords[1]} {atom_coords[2]} {atom_coords[3]}\n'
        return molecule.ind_from_xyz_string(xyz_string)
