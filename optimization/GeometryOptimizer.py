import os.path
import traceback
import uuid
from typing import Optional, List

import mlatom as ml
from mlatom.models import methods

from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from utils.FileUtils import mkdtemp_watchdog, free_watchdog_dir, get_files_with_suffix, get_dir_from_exception
from utils.XTBOptimizationInterface import xtb_optimization


class GeometryOptimizer:
    debug: bool
    optimization_program: str
    optimization_algorithm: str
    xtb_optimization_level: str
    convergence_criterion_for_forces: float
    method: str
    method_program: str
    config_path: Optional[str]
    working_directory: Optional[str]

    optimizations: int = 0
    failed_optimizations: int = 0
    evals: int = 0
    failed_evals: int = 0

    project_dir: str
    temp_config_path: str
    temp_opt_config_path: str

    def __init__(self, method: str, method_program: str, optimization_program: str = None, debug: bool = False,
                 optimization_algorithm: str = None, config_path: Optional[str] = None, working_directory: str = None,
                 convergence_criterion_for_forces: Optional[float] = None, project_dir: Optional[str] = None,
                 xtb_optimization_level: Optional[str] = None):
        self.optimization_program = optimization_program
        self.method = method
        self.method_program = method_program
        self.optimization_algorithm = optimization_algorithm
        self.convergence_criterion_for_forces = convergence_criterion_for_forces
        self.working_directory = working_directory
        self.debug = debug
        self.xtb_optimization_level = xtb_optimization_level
        self.config_path = config_path

        self.project_dir = project_dir
        self.temp_config_path = f'{self.working_directory}/xtb_config_temp'
        self.temp_opt_config_path = f'{self.working_directory}/xtb_opt_config_temp'

        if self.optimization_program is not None and self.optimization_program.casefold() == 'native':
            if self.method_program.casefold().strip() != 'xtb':
                raise RuntimeError('Native optimization program is only supported for XTB')
            self.set_xtb_opt_config(method, config_path)
        elif method_program.casefold().strip() == 'xtb':
            self.set_xtb_config(method, config_path)


    def optimize_geometry(self, mol: molecule, threads: int, max_steps: int,
                          temp_path: str = 'temp') -> Optional[molecule]:
        if self.optimization_program.casefold() == 'native':
            return self.__native_optimize(mol, threads, temp_path, self.config_path)
        match self.method_program.casefold():
            case 'orca':
                return self.__temp_optimize(mol, threads, max_steps, temp_path)
            case _:
                return self.__optimize(mol, threads, max_steps, self.config_path)


    def __native_optimize(self, mol: molecule, threads: int, temp_path: str, config_path: Optional[str] = None):
        if self.method_program.casefold() != 'xtb':
            raise RuntimeError('Native optimization program is only supported for XTB')
        temp_dir, observer = mkdtemp_watchdog(f'{self.working_directory}/{temp_path}')
        try:
            optimized, stdout, stderr = xtb_optimization(mol, self.project_dir, threads, temp_dir, config_path)
            if optimized is None and self.debug is True:
                self.__save_debug_file(temp_dir, f'{self.method_program}-xtb-opt', f'{stdout}\n{stderr}')
                return None
            optimized = self.__xtb_calculate(optimized, threads, False, self.temp_config_path)
        except Exception as e:
            if self.debug:
                self.__save_debug_file(temp_dir, f'{self.method_program}-xtb-opt', traceback.format_exc())
            return None
        finally:
            if temp_dir is not None:
                free_watchdog_dir(temp_dir, observer)
        return optimized

    def __temp_optimize(self, mol: molecule, threads: int, max_steps: int, temp_path: str) -> Optional[molecule]:
        temp_dir, observer = mkdtemp_watchdog(f'{self.working_directory}/{temp_path}')
        try:
            optimize_model = methods(method=self.method, program=self.method_program, nthreads=threads,
                                     working_directory=temp_dir)
            geom_opt = ml.optimize_geometry(model=optimize_model,
                                            initial_molecule=mol,
                                            working_directory=temp_dir,
                                            program=self.optimization_program,
                                            optimization_algorithm=self.optimization_algorithm,
                                            maximum_number_of_steps=max_steps,
                                            convergence_criterion_for_forces=self.convergence_criterion_for_forces)
        except Exception as e:
            if self.debug:
                self.__save_debug_file(temp_dir, f'{self.method_program}-eval', traceback.format_exc())
            return None
        finally:
            if temp_dir is not None:
                free_watchdog_dir(temp_dir, observer)
        return geom_opt.optimized_molecule

    def __optimize(self, mol: molecule, threads: int, max_steps: int, config_path: str = None) -> Optional[molecule]:
        if config_path is None:
            optimize_model = methods(method=self.method, program=self.method_program, nthreads=threads)
        else:
            optimize_model = methods(method=self.method, program=self.method_program, nthreads=threads,
                                     read_keywords_from_file=config_path)
        try:
            geom_opt = ml.optimize_geometry(model=optimize_model,
                                            initial_molecule=mol,
                                            program=self.optimization_program,
                                            optimization_algorithm=self.optimization_algorithm,
                                            maximum_number_of_steps=max_steps,
                                            convergence_criterion_for_forces=self.convergence_criterion_for_forces)
        except Exception as e:
            if self.debug:
                temp_dir: Optional[str] = get_dir_from_exception(e)
                self.__save_debug_file(temp_dir, f'{self.method_program}-opt', traceback.format_exc())
            return None
        return geom_opt.optimized_molecule

    def calculate_energy(self, mol: molecule, threads: int, xtb_without_d4: bool = False,
                         temp_path: str = 'temp') -> Optional[molecule]:
        match self.method_program.casefold():
            case 'orca':
                return self.__orca_calculate(mol, threads, temp_path)
            case 'pyscf':
                return self.__pyscf_calculate(mol, threads)
            case 'xtb':
                return self.__xtb_calculate(mol, threads, xtb_without_d4, self.config_path)
            case _:
                return self.__default_calculate(mol, threads)

    def __default_calculate(self, mol: molecule, threads: int) -> Optional[molecule]:
        print(f'Warning: using unknown program {self.method_program} (using defaults)')
        energy_model = methods(method=self.method, program=self.method_program, nthreads=threads)
        try:
            energy_model.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True,
                                 calculate_hessian=False)
        except Exception as e:
            if self.debug:
                temp_dir: Optional[str] = get_dir_from_exception(e)
                self.__save_debug_file(temp_dir, 'default-eval', traceback.format_exc())
            return None
        return mol

    def __xtb_calculate(self, mol: molecule, threads: int, without_d4: bool = False,
                        config_path: Optional[str] = None) -> Optional[molecule]:
        if config_path is None:
            energy_model = methods(method=self.method, program=self.method_program, nthreads=threads,
                                   without_d4=without_d4)
        else:
            energy_model = methods(method=self.method, program=self.method_program, nthreads=threads,
                                   without_d4=without_d4, read_keywords_from_file=config_path)
        try:
            energy_model.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True,
                                 calculate_hessian=False)
        except Exception as e:
            if self.debug:
                temp_dir: Optional[str] = get_dir_from_exception(e)
                self.__save_debug_file(temp_dir, f'{self.method_program}-eval', traceback.format_exc())
            return None
        return mol

    def __pyscf_calculate(self, mol: molecule, threads: int) -> Optional[molecule]:
        energy_model = methods(method=self.method, program=self.method_program, nthreads=threads)
        try:
            energy_model.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True,
                                 calculate_hessian=False)
        except Exception as e:
            if self.debug:
                temp_dir: Optional[str] = get_dir_from_exception(e)
                self.__save_debug_file(temp_dir, f'{self.method_program}-eval', traceback.format_exc())
            return None
        return mol


    def __orca_calculate(self, mol: molecule, threads: int, temp_path: str) -> Optional[molecule]:
        temp_dir, observer = mkdtemp_watchdog(f'{self.working_directory}/{temp_path}')

        energy_model = methods(method=self.method, program=self.method_program, nthreads=threads,
                               working_directory=temp_dir)
        try:
            energy_model.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True,
                                 working_directory=temp_dir, calculate_hessian=False)
        except Exception as e:
            if self.debug:
                self.__save_debug_file(temp_dir, f'{self.method_program}-eval', traceback.format_exc())
            return None
        finally:
            if temp_dir is not None:
                free_watchdog_dir(temp_dir, observer)
        return mol

    def __save_debug_file(self, temp_path: Optional[str], prefix: str, debug_msg: Optional[str] = None) -> None:
        debug_dir: str = f'{self.working_directory}/debug'
        if not os.path.exists(debug_dir):
            return

        if temp_path is not None and os.path.isdir(temp_path):
            inputs: List[str] = get_files_with_suffix(temp_path, '.inp')
            outputs: List[str] = get_files_with_suffix(temp_path, '.out')
        else:
            inputs: List[str] = []
            outputs: List[str] = []

        filename: str = f'{prefix}_{str(uuid.uuid4())}'
        with open(os.path.join(debug_dir, filename), 'w') as debug_file:
            if debug_msg is not None:
                debug_file.write(f'Exception:\n{debug_msg}\n\n---\n\n')
            try:
                if inputs is not None and len(inputs) > 0:
                    debug_file.write(f'Input files:\n\n')
                    for input_name in inputs:
                        debug_file.write(f'Input filename: "{input_name}":\n')
                        with open(input_name, 'r') as f:
                            debug_file.write(f.read())
                        debug_file.write(f'\n\n---\n\n')
                if outputs is not None and len(outputs) > 0:
                    debug_file.write(f'Output files:\n\n')
                    for output_name in outputs:
                        debug_file.write(f'Output filename: "{output_name}":\n')
                        with open(output_name, 'r') as f:
                            debug_file.write(f.read())
                        debug_file.write(f'\n\n---\n\n')
            except:
                pass

    def set_xtb_config(self, method: str, config_path: Optional[str] = None) -> None:
        if method.casefold().strip() in ['gfn1-xtb', 'xtb1', 'gfn1']:
            method_arg = '--gfn 1'
        elif method.casefold().strip() in ['gfn2-xtb', 'xtb2', 'gfn2']:
            method_arg = '--gfn 2'
        else:
            raise NotImplementedError(f'XTB method "{method}" is not recognized.')
        with open(self.temp_config_path, 'w') as f:
            if config_path is None:
                f.write(method_arg)
            else:
                with open(config_path, 'r') as conf:
                    original_config = conf.readline()
                if '--gfn' in original_config.casefold():
                    import re
                    new_config = re.sub(
                        r'--gfn \d',
                        method_arg,
                        original_config,
                        flags=re.IGNORECASE
                    )
                else:
                    new_config = original_config.rstrip() + ' ' + method_arg
                f.write(new_config)
        self.config_path = self.temp_config_path

    def set_xtb_opt_config(self, method: str, config_path: Optional[str] = None) -> None:
        if method.casefold().strip() in ['gfn1-xtb', 'xtb1', 'gfn1']:
            method_arg = '--gfn 1'
        elif method.casefold().strip() in ['gfn2-xtb', 'xtb2', 'gfn2']:
            method_arg = '--gfn 2'
        else:
            raise NotImplementedError(f'XTB method "{method}" is not recognized.')

        opt_arg = f'--opt {self.xtb_optimization_level}'
        with open(self.temp_opt_config_path, 'w') as f:
            if config_path is None:
                f.write(f'{method_arg} {opt_arg}')
            else:
                with open(config_path, 'r') as conf:
                    original_config = conf.readline()
                if '--gfn' in original_config.casefold():
                    import re
                    new_config = re.sub(
                        r'--gfn \d',
                        method_arg,
                        original_config,
                        flags=re.IGNORECASE
                    )
                else:
                    new_config = f'{original_config.rstrip()} {method_arg} {opt_arg}'
                f.write(new_config)
        self.config_path = self.temp_opt_config_path

    def equals(self, other: 'GeometryOptimizer') -> bool:
        if other is None:
            return False
        return self is other or (self.method == other.method and self.method_program == other.method_program)

    def update_eval_stats(self, stats: List[int]):
        self.evals = stats[0]
        self.failed_evals = stats[1]

    def update_optimization_stats(self, stats: List[int]):
        self.optimizations = stats[0]
        self.failed_optimizations = stats[1]
