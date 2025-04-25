import os
from typing import Optional

from mlatom.interfaces.xtb_interface import xtb_methods
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule



def xtb_optimization(mol: molecule, project_dir: str, threads: int, temp_dir: str, config_path: Optional[str] = None):
    xtb_interface = xtb_methods()

    if project_dir is None:
        xyz_filename = os.path.abspath(f'{temp_dir}/unoptimized0.xyz')
    else:
        xyz_filename = os.path.join(project_dir, f'{temp_dir}/unoptimized0.xyz')
    mol.write_file_with_xyz_coordinates(filename=xyz_filename)

    try:
        xtb_bin = os.environ['xtb']
    except:
        xtb_bin = "%s/xtb" % os.path.dirname(__file__)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    xtb_args = [xtb_bin, xyz_filename]
    if mol.charge != 0: xtb_args += ['-c', '%d' % mol.charge]
    number_of_unpaired_electrons = mol.multiplicity - 1
    xtb_args += ['-u', '%d' % number_of_unpaired_electrons]

    if config_path is not None:
        with open(config_path, 'r') as f:
            config_args = f.readline()
        xtb_args += config_args.split()

    terminated = False
    while not terminated:
        stdout, stderr = xtb_interface.run_xtb_job(xtb_args, temp_dir)
        rerun, terminated = xtb_interface.error_handle(stdout + stderr)
    xtb_scf_successful = not rerun

    if xtb_scf_successful:
        final_structure_start = -1
        for i in range(len(stdout)):
            if 'final structure:' in stdout[i].casefold():
                final_structure_start = i
                break
        if final_structure_start == -1:
            return None, stdout, stderr
        final_structure_start += 2
        atom_count = int(stdout[final_structure_start].strip())
        final_structure_start += 2 # comment

        xyz_string = f'{atom_count}\n\n'
        for i in range(atom_count):
            xyz_string += f'{stdout[final_structure_start + i].strip()}\n'

        return molecule.from_xyz_string(xyz_string), None, None
    return None, stdout, stderr