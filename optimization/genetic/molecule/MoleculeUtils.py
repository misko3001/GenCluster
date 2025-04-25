import shutil
import tempfile
from random import uniform
from typing import List, Tuple, Optional

import numpy as np
import rmsd
from MDAnalysis import Universe, AtomGroup
from rdkit import Chem
from rdkit.Chem import Mol, rdMolAlign
from numpy import ndarray
from scipy.linalg import eigvalsh
from scipy.optimize import linear_sum_assignment


AlignmentMethods = ['brute_force', 'rdkit', 'hungarian_rdkit', 'inertia_hungarian_rdkit']
RMSDMethods = AlignmentMethods + ['eigen']


def get_universe_from_molecule(mol, temp_path: str = 'data/temp', delete_file: bool = True) -> tuple[Universe, str]:
    temp_dir: str = tempfile.mkdtemp(dir=temp_path)
    file_path = f'{temp_dir}/molecule.xyz'
    mol.write_file_with_xyz_coordinates(file_path)
    universe = Universe(topology=file_path, topology_format='xyz', in_memory=True)
    if delete_file:
        shutil.rmtree(temp_dir)
    return universe, temp_dir


def get_molecule_from_universe(uv: Universe):
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
    xyz_string: str = universe_to_xyz_string(uv)
    return molecule.ind_from_xyz_string(xyz_string)


def universe_to_xyz_string(uv: Universe) -> str:
    atoms: AtomGroup = uv.atoms
    coords: ndarray = atoms.positions
    elements: List[str] = atoms.elements
    xyz_string: str = f'{atoms.n_atoms}\n\n'
    for i in range(atoms.n_atoms):
        xyz_string += f'{elements[i]} {coords[i][0]} {coords[i][1]} {coords[i][2]}\n'
    return xyz_string


def get_molecule_from_coords(coords: ndarray, element_symbols: List[str] | ndarray):
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule

    if element_symbols is ndarray:
        element_symbols = element_symbols.tolist()

    xyz_string: str = f'{coords.shape[0]}\n\n'
    for i in range(coords.shape[0]):
        xyz_string += f'{element_symbols[i]} {coords[i][0]} {coords[i][1]} {coords[i][2]}\n'
    return molecule.ind_from_xyz_string(xyz_string)


def element_coords_to_molecule(coords: List[ndarray], element_symbols: List[str]):
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
    xyz_string = f"{len(coords)}\n\n"
    for i, atom in enumerate(coords):
        coord = atom
        while True:
            if isinstance(coord[0], float):
                break
            coord = coord[0]
        xyz_string += f"{element_symbols[i]} {coord[0]} {coord[1]} {coord[2]}\n"
    return molecule.ind_from_xyz_string(xyz_string)


def generate_random_xyz_string(element: str, size: int, min_coords: float = -1, max_coords: float = 1) -> str:
    xyz = f'{size}\n\n'
    for _ in range(size):
        xyz += f'{element} {uniform(min_coords, max_coords)} {uniform(min_coords, max_coords)} {uniform(min_coords, max_coords)}'
    return xyz


def get_molecule_centroid(coordinates: ndarray) -> ndarray:
    return coordinates.mean(axis=0)


def get_molecule_center_of_mass(coordinates: ndarray, masses: ndarray) -> ndarray:
    return np.sum(coordinates * masses[:, np.newaxis], axis=0) / np.sum(masses)


def get_angstrom_distance(coords1: ndarray, coords2: ndarray) -> np.floating:
    return np.linalg.norm(coords1 - coords2)


def get_distances_and_adjacency_matrix(coords: ndarray, element_symbols: list[str],
                                       internuclear_distances: any) -> Tuple[ndarray, ndarray]:
    distance_rows: List[List[np.floating]] = []
    adjacency_rows: List[List[int]] = []
    for i in range(coords.shape[0]):
        distance_row: List[np.floating] = []
        adjacency_row: List[int] = []
        for j in range(coords.shape[0]):
            if i != j:
                distance: np.floating = get_angstrom_distance(coords[i], coords[j])
                distance_row.append(distance)
                symbol1, symbol2 = element_symbols[i], element_symbols[j]
                bond_distance = internuclear_distances.get(symbol1, symbol2)
                if bond_distance is None:
                    adjacency_row.append(0)
                else:
                    adjacency_row.append(1 if distance <= bond_distance[1] else 0)
            else:
                distance_row.append(0.)
                adjacency_row.append(0)
        distance_rows.append(distance_row)
        adjacency_rows.append(adjacency_row)
    distances: ndarray = np.array(distance_rows)
    adjacency_matrix: ndarray = np.array(adjacency_rows)
    return distances, adjacency_matrix


def check_min_distances(distances: ndarray, element_symbols: list[str],
                        internuclear_distances: any) -> Tuple[bool, Optional[int], Optional[int]]:
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if i != j:
                symbol1, symbol2 = element_symbols[i], element_symbols[j]
                bond_distance = internuclear_distances.get(symbol1, symbol2)
                if bond_distance is not None and distances[i][j] < bond_distance[0]:
                    return False, i, j
    return True, None, None


def eigenvalues_rmsd(mol1, mol2) -> float:
    """
    Compute the RMSD between two internuclear distance matrices based on their eigenvalues.

    Args:
        mol1, mol2: (molecule) Molecules with distance matrix of type (NxN).

    Returns:
        float: RMSD between the eigenvalue spectra.
    """
    distances1, distances2 = mol1.get_internuclear_distance_matrix(), mol2.get_internuclear_distance_matrix()
    eig1 = eigvalsh(distances1)
    eig2 = eigvalsh(distances2)
    return np.sqrt(np.mean((eig1 - eig2) ** 2))


def inertia_hungarian_reorder(target_molecule, reference_molecule):
    """
    Reorders the atoms of the target molecule to match the reference molecule
    using the inertia Hungarian algorithm.

    Args:
        target_molecule (MoleculeIndividual): Molecule to be reordered.
        reference_molecule (MoleculeIndividual): Molecule to be used a reference for reordering.

    Returns:
        MoleculeIndividual: Reordered molecule based on the optimal pairing of atoms from target to reference.
    """
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
    mol1: molecule = target_molecule
    mol2: molecule = reference_molecule

    mol1_coords: ndarray = mol1.get_xyz_coordinates()
    mol1_atoms: ndarray = mol1.get_atomic_numbers()
    mol2_coords: ndarray = mol2.get_xyz_coordinates()
    mol2_atoms: ndarray = mol2.get_atomic_numbers()

    mol1_coords -= rmsd.centroid(mol1_coords)
    mol2_coords -= rmsd.centroid(mol2_coords)

    indices = rmsd.reorder_inertia_hungarian(mol1_atoms, mol2_atoms, mol1_coords, mol2_coords)
    reordered_mol1_atoms: ndarray = mol1_atoms[indices]
    reordered_mol1_coords: ndarray = mol1_coords[indices]

    return get_molecule_from_coords(reordered_mol1_coords, reordered_mol1_atoms)


def hungarian_reorder(target_molecule, reference_molecule):
    """
    Solve the linear assignment problem (Hungarian algorithm) to minimize pairwise distances.

    Args:
        target_molecule (MoleculeIndividual): Molecule to be reordered.
        reference_molecule (MoleculeIndividual): Molecule to be used a reference for reordering.

    Returns:
        MoleculeIndividual: Reordered molecule based on the optimal pairing of atoms from target to reference.
    """
    unaligned_coords = target_molecule.get_xyz_coordinates()
    unaligned_element_symbols: List[str] = target_molecule.get_element_symbols().tolist()
    reference_coords = reference_molecule.get_xyz_coordinates()

    # Calculate pairwise distance matrix
    distance_matrix = np.linalg.norm(unaligned_coords[:, np.newaxis] - reference_coords, axis=2)

    # Hungarian algorithm (linear sum assignment)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    reordered_coords = unaligned_coords[col_ind]
    reordered_symbols: List[str] = [unaligned_element_symbols[i] for i in col_ind]

    return get_molecule_from_coords(reordered_coords, reordered_symbols)


def transform_rdkit_molecule(mol: Mol, transform, atom_map = None) -> Mol:
    """
    Transforms rdkit molecule based on the provided rotation matrix and translation vector.
    Optionally, reorders the atoms based on the atom map.
    """
    if atom_map is not None:
        new_order = [pair[0] for pair in sorted(atom_map, key=lambda x: x[1])]
        mol = Chem.RenumberAtoms(mol, new_order)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    rotation_matrix = transform[:3, :3]
    translation_vector = transform[:3, 3]
    new_positions = np.dot(positions, rotation_matrix.T) + translation_vector
    for i, pos in enumerate(new_positions):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*pos))
    return mol


def brute_force_alignment(mol1, mol2, threads: int = 1) -> Tuple[any, float]:
    """
    Returns aligned mol1 with minimal rmsd by checking ALL (!) the permutations
    """
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
    unaligned_mol1: Mol = Chem.MolFromXYZBlock(mol1.get_xyz_string())
    reference_mol2: Mol = Chem.MolFromXYZBlock(mol2.get_xyz_string())

    rmsd_value, transform, atom_map = rdMolAlign.GetBestAlignmentTransform(prbMol=unaligned_mol1,
                                                                           refMol=reference_mol2,
                                                                           numThreads=threads)

    aligned_mol = transform_rdkit_molecule(unaligned_mol1, transform)
    return molecule.ind_from_xyz_string(Chem.MolToXYZBlock(aligned_mol)), rmsd_value


def rdkit_alignment(mol1, mol2, max_iter: int = 500) -> Tuple[any, float]:
    """
    Returns aligned mol1 with minimal rmsd by using rdkit alignment.
    """
    from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
    unaligned_mol1: Mol = Chem.MolFromXYZBlock(mol1.get_xyz_string())
    reference_mol2: Mol = Chem.MolFromXYZBlock(mol2.get_xyz_string())

    rmsd_value, transform = rdMolAlign.GetAlignmentTransform(prbMol=unaligned_mol1,
                                                             refMol=reference_mol2,
                                                             maxIters=max_iter)

    unaligned_mol1 = transform_rdkit_molecule(unaligned_mol1, transform)
    return molecule.ind_from_xyz_string(Chem.MolToXYZBlock(unaligned_mol1)), rmsd_value


def hungarian_rdkit_alignment(mol1, mol2, max_iter: int = 500) -> Tuple[any, float]:
    """
    Returns aligned mol1 with minimal rmsd by using rdkit and hungarian algorithm.
    """
    reordered_molecule = hungarian_reorder(mol1, mol2)
    return rdkit_alignment(reordered_molecule, mol2, max_iter)


def inertia_hungarian_rdkit_alignment(mol1, mol2, max_iter: int = 500) -> Tuple[any, float]:
    """
    Returns aligned mol1 with minimal rmsd by using rdkit and inertia hungarian algorithm.
    """
    reordered_molecule = inertia_hungarian_reorder(mol1, mol2)
    return rdkit_alignment(reordered_molecule, mol2, max_iter)
