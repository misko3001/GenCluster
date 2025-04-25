import os
from datetime import datetime
from os.path import isfile
from typing import Optional, Tuple

from core.ConfigParser import AppConfig, load_app_config, InvalidConfiguration, MutatorConfig, TerminatorConfig
from core.InternuclearDistances import InternuclearDistances
from optimization.GeometryOptimizer import GeometryOptimizer
from optimization.genetic.EvolutionEngine import EvolutionEngine, EngineParameters
from optimization.genetic.molecule.MoleculeGenerator import MoleculeGenerator, PackMolGenerator, RandomGeometryGenerator
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual
from optimization.genetic.operations.Crossover import Crossover, TwoPointCrossover, SpliceCrossover
from optimization.genetic.operations.Mutator import Mutator, CreepMutator, TwinningMutator
from optimization.genetic.operations.Selector import Selector, EliteSelector, TournamentSelector, RmsdTournamentSelector
from optimization.genetic.operations.Terminator import Terminator, MaxIterationsTerminator, StagnationTerminator, \
    DurationTerminator, ConvergenceTerminator
from utils.FileUtils import is_absolute_path

default_working_directory = 'data/out'
default_optimize_every_nth_gen = 1


class EvolutionContext:
    config: AppConfig
    input_molecule: any
    project_directory: str
    internuclear_distances: InternuclearDistances
    optimizer: Optional[GeometryOptimizer]
    evaluator: Optional[GeometryOptimizer]
    generator: MoleculeGenerator
    selector: Selector
    elite_selector: Optional[EliteSelector]
    crossover: Crossover
    mutators: [Mutator]
    terminators: [Terminator]

    def __init__(self, yaml_config_path: str, working_directory: Optional[str] = None,
                 project_directory: Optional[str] = None):
        if not os.path.isfile(yaml_config_path):
            raise InvalidConfiguration(f'Invalid configuration path: "{yaml_config_path}"')
        config: AppConfig = load_app_config(yaml_config_path, project_directory)
        self.config = config
        self.__set_defaults(working_directory, project_directory)
        self.generator = self.__build_generator(config, self.internuclear_distances)
        self.selector = self.__build_selector(config)
        self.elite_selector = self.__build_elite_selector(config)
        self.crossover = self.__build_crossover(config, self.input_molecule, self.internuclear_distances)
        self.mutators = self.__build_mutators(config)
        self.terminators = self.__build_terminators(config)

    def __set_defaults(self, working_directory: Optional[str] = None, project_directory: Optional[str] = None):
        problem_config = self.config.problem
        self.project_directory = project_directory
        if project_directory is not None:
            optimizer_config = self.config.optimizer
            evaluator_config = self.config.evaluator
            population_config = self.config.population
            self.__make_absolute_paths(problem_config, optimizer_config, evaluator_config, project_directory, population_config)
        if working_directory is not None:
            problem_config.working_directory = working_directory
            print(f'Overriding the working directory to "{working_directory}"')
        elif problem_config.working_directory is None:
            output_dir = datetime.now().__str__().replace(':', '-').replace(' ', '_')
            problem_config.working_directory = f'{default_working_directory}/{output_dir}'
        self.input_molecule = MoleculeIndividual.from_xyz_file(problem_config.input_molecule_path)
        self.input_molecule.charge = problem_config.molecule_charge
        self.input_molecule.multiplicity = problem_config.molecule_multiplicity
        self.internuclear_distances = InternuclearDistances(self.input_molecule,
                                                            problem_config.internuclear_angstrom_distances)
        if problem_config.alignment_methods is None:
            problem_config.alignment_methods = ['rdkit', 'hungarian_rdkit', 'inertia_hungarian_rdkit']

        if problem_config.rmsd_methods is None:
            problem_config.rmsd_methods = ['eigen']

        if self.config.optimizer is not None and self.config.optimizer.XTB_optimization_level is None:
            self.config.optimizer.XTB_optimization_level = 'extreme'

    def get_evolution_engine(self):
        self.optimizer = self.__build_optimizer(self.config, self.project_directory)
        self.evaluator = self.__build_evaluator(self.config)

        max_optimization_steps = None
        optimization_processes = None
        optimization_threads = None
        optimize_every_nth_generation = None
        optimize_n_best = None
        optimize_rmsd_distinct_threshold = None
        initial_optimization = False
        max_optimization_threads = None
        max_evaluation_threads = None
        min_unoptimized_gradient_norm = None

        if self.config.evaluator is not None:
            max_evaluation_threads = self.config.evaluator.model.max_threads

        if self.config.optimizer is not None:
            max_optimization_steps = self.config.optimizer.max_optimization_steps
            optimization_processes = self.config.optimizer.processes
            optimization_threads = self.config.optimizer.model.threads
            max_optimization_threads = self.config.optimizer.model.max_threads
            initial_optimization = self.config.optimizer.initial_optimization
            min_unoptimized_gradient_norm = self.config.optimizer.args.min_unoptimized_gradient_norm
            if self.config.evaluator is None:
                max_evaluation_threads = self.config.optimizer.model.max_threads
            if self.config.optimizer.args is not None:
                optimize_every_nth_generation = self.config.optimizer.args.optimize_every_nth_generation
                optimize_n_best = self.config.optimizer.args.optimize_n_best
                optimize_rmsd_distinct_threshold = self.config.optimizer.args.optimize_rmsd_distinct_threshold

        engine_params: EngineParameters = EngineParameters(
            max_optimization_steps=max_optimization_steps,
            optimization_processes=optimization_processes,
            optimization_threads=optimization_threads,
            evaluator_processes=self.config.evaluator.processes if self.config.evaluator is not None else optimization_processes,
            evaluator_threads=self.config.evaluator.model.threads if self.config.evaluator is not None else optimization_threads,
            population_size=self.config.population.size,
            max_age=self.config.population.max_age,
            cluster_size=self.config.problem.cluster_size,
            input_molecule=self.input_molecule,
            working_directory=self.config.problem.working_directory,
            checkpoint_frequency=self.config.problem.checkpoint_every_nth_generation,
            optimize_n_best=optimize_n_best,
            optimize_rmsd_distinct_threshold=optimize_rmsd_distinct_threshold,
            optimize_every_nth_generation=optimize_every_nth_generation,
            internuclear_distances=self.internuclear_distances,
            leaderboard_size=self.config.problem.leaderboard_size,
            leaderboard_rmsd_threshold=self.config.problem.leaderboard_rmsd_threshold,
            alignment_methods=self.config.problem.alignment_methods,
            rmsd_methods=self.config.problem.rmsd_methods,
            initial_optimization=initial_optimization,
            max_optimization_threads=max_optimization_threads,
            max_evaluation_threads=max_evaluation_threads,
            min_unoptimized_gradient_norm=min_unoptimized_gradient_norm,
            initial_guess_path=self.config.population.initial_guess_path
        )

        return EvolutionEngine(optimizer=self.optimizer,
                               evaluator=self.evaluator,
                               generator=self.generator,
                               selector=self.selector,
                               crossover=self.crossover,
                               mutators=self.mutators,
                               terminators=self.terminators,
                               elite=self.elite_selector,
                               params=engine_params)

    def __build_terminators(self, config: AppConfig) -> [Terminator]:
        terminators: [Terminator] = []
        for terminator_entry in config.terminators:
            terminator_config: TerminatorConfig = next(iter(terminator_entry.values()))
            terminators.append(self.__build_terminator(terminator_config))
        return terminators

    @staticmethod
    def __build_terminator(config: TerminatorConfig) -> Terminator:
        if config.name == 'MaxIterationsTerminator':
            return MaxIterationsTerminator(config.args.max_iterations)
        elif config.name == 'StagnationTerminator':
            return StagnationTerminator(config.args.max_stagnating_iterations)
        elif config.name == 'DurationTerminator':
            return DurationTerminator(config.args.unit_type, config.args.unit_value, datetime.now())
        elif config.name == 'ConvergenceTerminator':
            return ConvergenceTerminator(config.args.last_m_generations, config.args.last_n_generations,
                                         config.args.delta)
        raise NotImplementedError(f'Terminator "{config.name}" is not implemented')

    def __build_mutators(self, config: AppConfig) -> [Mutator]:
        mutators: [Mutator] = []
        for mutator_entry in config.mutators:
            mutator_config: MutatorConfig = next(iter(mutator_entry.values()))
            mutators.append(self.__build_mutator(mutator_config, self.input_molecule, self.internuclear_distances))
        return mutators

    @staticmethod
    def __build_mutator(config: MutatorConfig, mol: any = None, internuclear_distances: any = None) -> Mutator:
        if config.name == 'CreepMutator':
            return CreepMutator(probability=config.args.probability,
                                per_atom_probability=config.args.per_atom_probability,
                                max_mutation_strength=config.args.max_mutation_strength)
        elif config.name == 'TwinningMutator':
            return TwinningMutator(probability=config.args.probability,
                                   base_mol_size=len(mol.atoms),
                                   internuclear_distances=internuclear_distances)
        raise NotImplementedError(f'Mutator "{config.name}" is not implemented')

    @staticmethod
    def __build_crossover(config: AppConfig, mol: any = None, internuclear_distances: any = None) -> Crossover:
        if config.crossover.name == 'TwoPointCrossover':
            return TwoPointCrossover(probability=config.crossover.args.probability,
                                     internuclear_distances=internuclear_distances,
                                     alignment_methods=config.problem.alignment_methods)
        elif config.crossover.name == 'SpliceCrossover':
            return SpliceCrossover(probability=config.crossover.args.probability,
                                   base_mol_size=len(mol.atoms),
                                   internuclear_distances=internuclear_distances)
        raise NotImplementedError(f'Crossover "{config.crossover.name}" is not implemented')

    @staticmethod
    def __build_elite_selector(config: AppConfig) -> Optional[EliteSelector]:
        if config.selector.elite_quantity is not None and config.selector.elite_quantity > 0:
            return EliteSelector(config.selector.elite_quantity)
        return None

    @staticmethod
    def __build_selector(config: AppConfig) -> Selector:
        if config.selector.name == 'TournamentSelector':
            return TournamentSelector(config.selector.args.tournament_size)
        elif config.selector.name == 'RmsdTournamentSelector':
            return RmsdTournamentSelector(config.problem.rmsd_methods, config.selector.args.tournament_size)
        raise NotImplementedError(f'Selector "{config.selector.name}" is not implemented')

    @staticmethod
    def __build_generator(config: AppConfig, internuclear_distances: InternuclearDistances) -> MoleculeGenerator:
        angstrom_distances: Tuple[float, float] = (config.generator.args.min_angstrom_distance,
                                                   config.generator.args.max_angstrom_distance)
        if config.generator.name == 'PackMolGenerator':
            return PackMolGenerator(com_angstrom_distances=angstrom_distances,
                                    internuclear_distances=internuclear_distances)
        elif config.generator.name == 'RandomGeometryGenerator':
            return RandomGeometryGenerator(com_angstrom_distances=angstrom_distances,
                                           internuclear_distances=internuclear_distances)
        raise NotImplementedError(f'Generator "{config.generator.name}" is not implemented')

    def __build_evaluator(self, config: AppConfig) -> GeometryOptimizer:
        if config.evaluator is not None:
            return GeometryOptimizer(method=config.evaluator.model.method,
                                     method_program=config.evaluator.model.program,
                                     working_directory=config.problem.working_directory,
                                     config_path=config.evaluator.model.config_path,
                                     debug=config.evaluator.debug)
        return self.optimizer

    @staticmethod
    def __build_optimizer(config: AppConfig, project_dir) -> Optional[GeometryOptimizer]:
        if config.optimizer is not None:
            opt_program: str = config.optimizer.optimization_program
            if opt_program.casefold().strip() == 'ase' and config.optimizer.ASE_optimization_algorithm is None:
                config.optimizer.ASE_optimization_algorithm = 'LBFGS'
            return GeometryOptimizer(optimization_program=opt_program,
                                     optimization_algorithm=config.optimizer.ASE_optimization_algorithm,
                                     method=config.optimizer.model.method,
                                     method_program=config.optimizer.model.program,
                                     working_directory=config.problem.working_directory,
                                     config_path=config.optimizer.model.config_path,
                                     debug=config.optimizer.debug,
                                     convergence_criterion_for_forces=config.optimizer.ASE_forces_convergence_criterion,
                                     xtb_optimization_level=config.optimizer.XTB_optimization_level,
                                     project_dir=project_dir)
        return None

    @staticmethod
    def __make_absolute_paths(problem_config, optimizer_config, evaluator_config, project_directory, population_config):
        if not is_absolute_path(problem_config.input_molecule_path):
            problem_config.input_molecule_path = project_directory + '/' + problem_config.input_molecule_path
            if not isfile(problem_config.input_molecule_path):
                raise RuntimeError(f'Invalid molecule path: {problem_config.input_molecule_path}')
        if (optimizer_config is not None and optimizer_config.model.config_path is not None and
                not is_absolute_path(optimizer_config.model.config_path)):
            optimizer_config.model.config_path = project_directory + '/' + optimizer_config.model.config_path
            if not isfile(optimizer_config.model.config_path):
                raise RuntimeError(f'Invalid optimizer config path: {optimizer_config.model.config_path}')
        if (evaluator_config is not None and evaluator_config.model.config_path is not None and
                not is_absolute_path(evaluator_config.model.config_path)):
            evaluator_config.model.config_path = project_directory + '/' + evaluator_config.model.config_path
            if not isfile(evaluator_config.model.config_path):
                raise RuntimeError(f'Invalid evaluator config path: {evaluator_config.model.config_path}')
        if (population_config is not None and population_config.initial_guess_path is not None and
                not is_absolute_path(population_config.initial_guess_path)):
            population_config.initial_guess_path = project_directory + '/' + population_config.initial_guess_path
            if not isfile(population_config.initial_guess_path):
                raise RuntimeError(f'Invalid evaluator config path: {evaluator_config.model.config_path}')

