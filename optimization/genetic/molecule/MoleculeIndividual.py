import sys
from typing import Optional, Union, List, Tuple

import networkx as nx
import numpy as np
from mlatom.data import molecule
from networkx.classes import Graph
from numpy import ndarray

from core.InternuclearDistances import InternuclearDistances
from optimization.genetic.molecule.MoleculeUtils import get_distances_and_adjacency_matrix, check_min_distances, \
    brute_force_alignment, rdkit_alignment, hungarian_rdkit_alignment, inertia_hungarian_rdkit_alignment, \
    eigenvalues_rmsd


class MoleculeIndividual(molecule):
    modified: bool = False
    age: int = 0
    rmsd: Optional[tuple[any, float]] = None

    def get_fitness(self, rmsd_reference: Optional[molecule] = None, rmsd: Optional[float] = None) -> Optional[float]:
        if not self.has_fitness():
            return None
        elif rmsd_reference is None and rmsd is None:
            return self.energy
        elif rmsd is not None:
            return self.energy * self.rmsd_penalty_function(rmsd)
        else:
            rmsd = self.get_rmsd(rmsd_reference)
            return self.energy * self.rmsd_penalty_function(rmsd)

    def has_fitness(self) -> bool:
        return hasattr(self, 'energy')

    def is_modified(self) -> bool:
        return self.modified

    def get_compare_fitness_value(self) -> float:
        if self.has_fitness():
            return self.get_fitness()
        else:
            return sys.float_info.max

    def mark_change(self):
        self.modified = True
        self.age = 0
        self.rmsd = None
        if self.has_fitness():
            del self.energy

    def align_with(self, reference: molecule, methods: List[str], threads: int = 1,
                   max_iter: int = 1000) -> 'MoleculeIndividual':
        """
        Returns aligned copy of self by using the requested method
        """
        if self is reference:
            aligned_self = self.copy()
            aligned_self.rmsd = reference.id, 0
            aligned_self.energy = self.get_fitness()
            return aligned_self

        aligned_self, rmsd = self.__align(reference, methods, threads, max_iter)

        aligned_self.rmsd = reference.id, rmsd
        aligned_self.age = self.age
        aligned_self.modified = self.modified
        if self.has_fitness():
            aligned_self.energy = self.get_fitness()

        return aligned_self

    def __align(self, reference: molecule, methods: List[str], threads: int,
                max_iter: int) -> tuple['MoleculeIndividual', float]:
        best_aligned: Optional['MoleculeIndividual'] = None
        best_rmsd: Optional[float] = None

        for method in methods:
            match method:
                case 'rdkit':
                    aligned_self, rmsd = rdkit_alignment(self, reference, max_iter)
                case 'hungarian_rdkit':
                    aligned_self, rmsd = hungarian_rdkit_alignment(self, reference, max_iter)
                case 'inertia_hungarian_rdkit':
                    aligned_self, rmsd = inertia_hungarian_rdkit_alignment(self, reference, max_iter)
                case 'brute_force':
                    aligned_self, rmsd = brute_force_alignment(self, reference, threads)
                case _:
                    raise ValueError(f'Unknown alignment method: {method}')

            if best_rmsd is None or rmsd < best_rmsd:
                best_aligned = aligned_self
                best_rmsd = rmsd

        return best_aligned, best_rmsd

    def __calculate_rmsd(self, reference: molecule, methods: List[str], threads: int, max_iter: int) -> float:
        best_rmsd: Optional[float] = None

        for method in methods:
            match method:
                case 'eigen':
                    rmsd = eigenvalues_rmsd(self, reference)
                case 'rdkit':
                    aligned_self, rmsd = rdkit_alignment(self, reference, max_iter)
                case 'hungarian_rdkit':
                    aligned_self, rmsd = hungarian_rdkit_alignment(self, reference, max_iter)
                case 'inertia_hungarian_rdkit':
                    aligned_self, rmsd = inertia_hungarian_rdkit_alignment(self, reference, max_iter)
                case 'brute_force':
                    aligned_self, rmsd = brute_force_alignment(self, reference, threads)
                case _:
                    raise ValueError(f'Unknown RMSD method: {method}')

            if best_rmsd is None or rmsd < best_rmsd:
                best_rmsd = rmsd

        return best_rmsd

    def get_rmsd(self, reference: molecule, methods: Optional[List[str]] = None) -> Optional[float]:
        if methods is None and self.rmsd is not None and self.rmsd[0] == reference.id:
            return self.rmsd[1]
        return self.__calculate_rmsd(reference, methods, 1, 1000)

    @staticmethod
    def rmsd_penalty_function(rmsd: Optional[float], max_rmsd: float = 0.5) -> float:
        if rmsd is None:
            return 1
        return min(1.0, rmsd / max_rmsd)

    def is_valid(self, internuclear_distances: InternuclearDistances) -> bool:
        symbols: List[str] = [atom.element_symbol for atom in self.atoms]
        xyz_coords = self.get_xyz_coordinates()
        distances, adjacencies = get_distances_and_adjacency_matrix(xyz_coords, symbols, internuclear_distances)
        return self.check_min_distances(distances, symbols, internuclear_distances) and self.is_connected(adjacencies)

    def has_valid_min_distances(self, internuclear_distances: InternuclearDistances) -> Tuple[bool, ndarray]:
        symbols: List[str] = [atom.element_symbol for atom in self.atoms]
        xyz_coords = self.get_xyz_coordinates()
        distances, adjacencies = get_distances_and_adjacency_matrix(xyz_coords, symbols, internuclear_distances)
        return self.check_min_distances(distances, symbols, internuclear_distances), distances

    def has_valid_connectedness(self, internuclear_distances: InternuclearDistances) -> Tuple[bool, ndarray]:
        symbols: List[str] = [atom.element_symbol for atom in self.atoms]
        xyz_coords = self.get_xyz_coordinates()
        distances, adjacencies = get_distances_and_adjacency_matrix(xyz_coords, symbols, internuclear_distances)
        return self.is_connected(adjacencies), adjacencies

    @staticmethod
    def check_min_distances(distances: ndarray, symbols: List[str],
                            internuclear_distances: InternuclearDistances) -> bool:
        return check_min_distances(distances, symbols, internuclear_distances)[0]

    @staticmethod
    def is_connected(adjacency_matrix: ndarray) -> bool:
        graph: Graph = nx.from_numpy_array(adjacency_matrix)
        return nx.is_connected(graph)

    @classmethod
    def ind_from_xyz_file(cls, filename: str, format: Union[str, None] = None) -> 'MoleculeIndividual':
        mol: cls = MoleculeIndividual()
        return mol.from_xyz_file(filename, format)

    @classmethod
    def ind_from_xyz_string(cls, string: str = None, format: Union[str, None] = None) -> 'MoleculeIndividual':
        mol: cls = MoleculeIndividual()
        return mol.from_xyz_string(string, format)

    def is_heteronuclear(self) -> bool:
        symbols = self.get_element_symbols().tolist()
        return len(set(symbols)) > 1

    def get_gradient_norm(self) -> Optional[float]:
        if self.energy_gradients is not None and not np.any(np.isnan(self.energy_gradients)):
            gradient_norm = np.linalg.norm(self.energy_gradients)
            return gradient_norm if not np.isnan(gradient_norm) else None
        return None

    def get_short_info(self) -> str:
        info: str = 'XYZ string:\n' + self.get_xyz_string() + '\n'
        if self.has_fitness():
            info += f'Energy: {self.get_fitness():.6f} Hartree\n'
        else:
            info += f'Energy: unknown\n'
        if self.energy_gradients is not None and not np.any(np.isnan(self.energy_gradients)):
            info += f'Energy gradients norm: {np.linalg.norm(self.energy_gradients):.6f} Hartree/Angstrom\n'
        else:
            info += f'Energy gradients norm: unknown\n'
        return info

