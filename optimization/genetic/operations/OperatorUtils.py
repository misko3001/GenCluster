from bisect import insort
from random import uniform
from typing import List, Tuple, Optional, Set

import networkx as nx
import numpy as np
from numpy import ndarray

from core.InternuclearDistances import InternuclearDistances
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.MoleculeUtils import get_distances_and_adjacency_matrix, check_min_distances, \
    get_molecule_center_of_mass

MolCenters = List[Tuple[ndarray, ndarray]]


def split_centers_and_pair(mol1: molecule, mol2: molecule, mol_size: int) -> Tuple[MolCenters, MolCenters]:
    mol1_group1, mol1_group2 = split_centers(mol1, mol_size)
    mol2_group1, mol2_group2 = split_centers(mol2, mol_size, len(mol1_group1))

    if len(mol1_group1) == len(mol2_group1):
        ch1_group = mol1_group2 + mol2_group1
        ch2_group = mol1_group1 + mol2_group2
    else:
        ch1_group = mol1_group1 + mol2_group1
        ch2_group = mol1_group2 + mol2_group2

    return ch1_group, ch2_group


def reconnect_heteronuclear(groups: List[ndarray], element_symbols: List[str], max_tries: int,
                            internuclear_distances: InternuclearDistances, base_size: int,
                            verbose: bool = False) -> Optional[List[ndarray]]:
    groups = reconnect_base_molecules(groups, element_symbols, max_tries, internuclear_distances, base_size, verbose)
    if groups is None:
        if verbose:
            print('Could not reconnect base molecules')
        return None
    return reconnect_group(groups, element_symbols, max_tries, internuclear_distances, verbose, base_size)


def reconnect_base_molecules(groups: List[ndarray], element_symbols: List[str], max_tries: int,
                            internuclear_distances: InternuclearDistances, base_size: int,
                            verbose: bool = False) -> Optional[List[ndarray]]:
    reconnected_groups: List[ndarray] = []
    for i in range(len(groups)):
        coords: List[ndarray] = [groups[i]]
        symbols: List[str] = element_symbols[i * base_size : (i + 1) * base_size]
        reconnected_base_mol = reconnect_group(coords, symbols, max_tries, internuclear_distances, verbose, base_size)
        if reconnected_base_mol is None:
            return None
        reconnected_groups.append(np.array(reconnected_base_mol))
    return reconnected_groups


def reconnect_group(group: List[ndarray], element_symbols: List[str], max_tries: int,
                    internuclear_distances: InternuclearDistances, verbose: bool = False,
                    base_size: Optional[int] = None) -> Optional[List[ndarray]]:
    coords: ndarray = np.array(group)
    distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                     internuclear_distances)
    reconnect_result = reconnect_components(group, coords, element_symbols, distances, internuclear_distances,
                                            adjacency_matrix, max_tries, base_size)
    if reconnect_result is None:
        return None
    group, coords, distances = reconnect_result

    min_constraint = check_min_distances(distances, element_symbols, internuclear_distances)
    reconnect_group_tries: int = 0
    while not min_constraint[0]:
        if reconnect_group_tries >= max_tries:
            if verbose:
                print(f"Unable to reconnect group in {max_tries} tries")
            return None

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

        if base_size is not None:
            group, coords = translate_base_molecule_by_atom_index(atom2_index, group, coords, translation_vector,
                                                                  base_size)
        else:
            coords[atom2_index] += translation_vector
            group[atom2_index] += translation_vector

        distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                         internuclear_distances)
        reconnect_atoms_result = reconnect_atoms(group, coords, element_symbols, distances, internuclear_distances,
                                                 adjacency_matrix, max_tries)
        if reconnect_atoms_result is None:
            return None
        distances, adjacency_matrix = reconnect_atoms_result

        min_constraint = check_min_distances(distances, element_symbols, internuclear_distances)
        reconnect_group_tries += 1

    return group


def reconnect_components(group: List[ndarray], coords: ndarray, element_symbols: List[str], distances: ndarray,
                         internuclear_distances: InternuclearDistances, adjacency_matrix: ndarray, max_tries: int,
                         base_size: Optional[int] = None,
                         verbose: bool = False) -> Optional[Tuple[List[ndarray], ndarray, ndarray]]:
    tries: int = 0
    adjacency_graph = nx.from_numpy_array(adjacency_matrix)
    components: List[Set[int]] = list(nx.connected_components(adjacency_graph))
    while len(components) > 1:
        tries += 1
        if tries >= max_tries:
            if verbose:
                print(f'Unable to reconnect components in {tries} tries: {components}')
            return None

        base_component = components[0]
        disconnected_component = components[1]

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
            if base_size is not None:
                group = reconnect_base_molecules(group, element_symbols, max_tries, internuclear_distances, base_size,
                                                 verbose)
                if group is None:
                    return None
                coords = np.array(group)
                continue
            if verbose:
                print(f"Could not form a bond between two components {base_component} and {disconnected_component}.")
            return None

        base_symbol: str = element_symbols[closest_pair[0]]
        disconnected_symbol: str = element_symbols[closest_pair[1]]
        allowed_distance: Tuple[float, float] = internuclear_distances.get(base_symbol, disconnected_symbol)

        current_distance = distances[closest_pair[0]][closest_pair[1]]
        target_distance = uniform(allowed_distance[0], allowed_distance[1])
        distance_to_move = current_distance - target_distance

        direction = coords[closest_pair[0]] - coords[closest_pair[1]]
        translation_vector = (direction / np.linalg.norm(direction)) * distance_to_move

        if base_size is not None:
            group, coords = translate_base_molecule_by_atom_index(closest_pair[1], group, coords, translation_vector,
                                                                  base_size)
        else:
            for disconnected_atom_index in disconnected_component:
                    group[disconnected_atom_index] += translation_vector
                    coords[disconnected_atom_index] += translation_vector

        distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                         internuclear_distances)
        adjacency_graph = nx.from_numpy_array(adjacency_matrix)
        components = list(nx.connected_components(adjacency_graph))
    return group, coords, distances


def translate_base_molecule_by_atom_index(atom_index: int, group: List[ndarray], coords: ndarray,
                                          translation_vector, base_size: int) -> Tuple[List[ndarray], ndarray]:
    base_mol_index: int = atom_index // base_size
    indices_start = base_mol_index * base_size
    indices_end = indices_start + base_size
    for i in range(indices_start, indices_end):
        group[i] += translation_vector
        coords[i] += translation_vector
    return group, coords


def reconnect_atoms(group: List[ndarray], coords: ndarray, element_symbols: List[str], distances: ndarray,
                    internuclear_distances: InternuclearDistances, adjacency_matrix: ndarray, max_tries: int,
                    verbose: bool = False, base_size: Optional[int] = None) -> Optional[Tuple[ndarray, ndarray]]:
    adjacency_graph = nx.from_numpy_array(adjacency_matrix)
    components: List[Set[int]] = list(nx.connected_components(adjacency_graph))

    reconnect_atoms_tries: int = 0
    while len(components) > 1:
        if reconnect_atoms_tries >= max_tries:
            if verbose:
                print(f"Unable to reconnect atoms in {max_tries} tries")
            return None

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
            if base_size is not None:
                group = reconnect_base_molecules(group, element_symbols, max_tries, internuclear_distances, base_size,
                                                 verbose)
                coords = np.array(group)
                continue
            else:
                raise RuntimeError(f"Could not form a bond between two components {base_component}"
                                   f" and {disconnected_component}.")

        base_symbol: str = element_symbols[closest_pair[0]]
        disconnected_symbol: str = element_symbols[closest_pair[1]]
        allowed_distance: Tuple[float, float] = internuclear_distances.get(base_symbol, disconnected_symbol)

        current_distance = distances[closest_pair[0]][closest_pair[1]]
        target_distance = uniform(allowed_distance[0], allowed_distance[1])
        distance_to_move = current_distance - target_distance

        direction = coords[closest_pair[0]] - coords[closest_pair[1]]
        translation_vector = (direction / np.linalg.norm(direction)) * distance_to_move

        if base_size is not None:
            group, coords = translate_base_molecule_by_atom_index(closest_pair[1], group, coords,
                                                                  translation_vector, base_size)
        else:
            group[closest_pair[1]] += translation_vector
            coords[closest_pair[1]] += translation_vector

        distances, adjacency_matrix = get_distances_and_adjacency_matrix(coords, element_symbols,
                                                                         internuclear_distances)
        adjacency_graph = nx.from_numpy_array(adjacency_matrix)
        components: List[Set[int]] = list(nx.connected_components(adjacency_graph))

        reconnect_atoms_tries += 1

    return distances, adjacency_matrix


def get_centers_of_mass(mol: molecule, mol_size: int) -> Tuple[ndarray, MolCenters]:
    centers: List[Tuple[ndarray, ndarray]] = []
    xyz_coords: ndarray = mol.get_xyz_coordinates()
    cluster_masses: ndarray = mol.get_nuclear_masses()
    cluster_center: ndarray = get_molecule_center_of_mass(xyz_coords, cluster_masses)
    num_chunks = xyz_coords.shape[0] // mol_size
    for i in range(num_chunks):
        coords = xyz_coords[i * mol_size:(i + 1) * mol_size]
        masses = cluster_masses[i * mol_size:(i + 1) * mol_size]
        mol_center: ndarray = get_molecule_center_of_mass(coords, masses)
        centers.append((mol_center, coords))
    return cluster_center, centers


def split_centers(mol: molecule, mol_size: int,
                  required_split: Optional[int] = None) -> Tuple[MolCenters, MolCenters]:
    cluster_center, centers = get_centers_of_mass(mol, mol_size)

    plane_normal = np.random.rand(3) - 0.5
    plane_normal /= np.linalg.norm(plane_normal)

    group1: MolCenters = []
    group2: MolCenters = []

    if required_split is None:
        for center in centers:
            signed_distance = np.dot(plane_normal, center[0] - cluster_center)
            if signed_distance >= 0:
                group1.append(center)
            else:
                group2.append(center)
    else:
        center_count = len(centers)
        if required_split > center_count:
            raise RuntimeError(f'Required split ({required_split}) larger than cluster size {len(centers)}')
        group1_center_count: int = 0

        split_centers: List[Tuple[float, Tuple[ndarray, ndarray]]] = []
        for center in centers:
            signed_distance = np.dot(plane_normal, center[0] - cluster_center)
            insort(split_centers, (signed_distance, center))
            if signed_distance >= 0:
                group1_center_count += 1

        group2_center_count: int = center_count - group1_center_count
        if required_split == group1_center_count or required_split == group2_center_count:
            for i in range(center_count):
                if i < group1_center_count:
                    group1.append(split_centers[i][1])
                else:
                    group2.append(split_centers[i][1])
        else:
            group1_missing = required_split - group1_center_count
            group2_missing = required_split - group2_center_count
            if abs(group1_missing) <= abs(group2_missing):
                group1 = [split_centers[i][1] for i in range(group1_center_count + group1_missing)]
                group2 = [
                    split_centers[i][1] for i in range(group1_center_count + group1_missing, center_count)
                ]
            else:
                group1 = [split_centers[i][1] for i in range(group1_center_count - group2_missing)]
                group2 = [
                    split_centers[i][1] for i in range(group1_center_count - group2_missing, center_count)
                ]

    return group1, group2


def unpack_centers(group1: MolCenters, group2: MolCenters) -> Tuple[List[ndarray], List[ndarray]]:
    modified_group1: List[ndarray] = []
    modified_group2: List[ndarray] = []
    for group in group1:
        for i in range(group[1].shape[0]):
            modified_group1.append(group[1][i])
    for group in group2:
        for i in range(group[1].shape[0]):
            modified_group2.append(group[1][i])
    return modified_group1, modified_group2


def random_rotation_matrix() -> ndarray:
    """
    Generate a random 3D rotation matrix using the Gram-Schmidt process.
    """
    rand_vectors = np.random.normal(size=(3, 3))  # Random 3x3 matrix
    q, _ = np.linalg.qr(rand_vectors)  # Orthogonalize to get rotation matrix
    if np.linalg.det(q) < 0:  # Ensure it's a proper rotation matrix
        q[:, 2] *= -1
    return q


def rotate_coordinates(coords: ndarray, rotation_matrix: ndarray) -> ndarray:
    """
    Rotate a set of 3D coordinates using the given rotation matrix.
    Args:
        coords: ndarray of shape (N, 3), where N is the number of points.
        rotation_matrix: ndarray of shape (3, 3), a 3D rotation matrix.
    Returns:
        Rotated coordinates as a ndarray of the same shape as `coords`.
    """
    return np.dot(coords, rotation_matrix.T)
