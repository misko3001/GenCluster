import time
from typing import Optional, List, Tuple

from joblib import Parallel, delayed

from core.Checkpoint import save_checkpoint, RestoredCheckpoint
from optimization.GeometryOptimizer import GeometryOptimizer
from optimization.genetic.molecule.MoleculeGenerator import MoleculeGenerator
from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.PopulationUtils import generate_population, get_best_individuals, \
    print_fitness, PopulationStats, load_initial_guesses
from optimization.genetic.operations.Crossover import Crossover
from optimization.genetic.operations.Mutator import Mutator
from optimization.genetic.operations.Selector import Selector, EliteSelector
from optimization.genetic.operations.Terminator import Terminator


class EngineParameters:
    max_optimization_steps: Optional[int]
    optimization_processes: Optional[int]
    optimization_threads: Optional[int]
    evaluator_processes: int
    evaluator_threads: int
    population_size: int
    max_age: Optional[int]
    cluster_size: int
    input_molecule: molecule
    working_directory: str
    checkpoint_frequency: Optional[int]
    optimize_every_nth_generation: Optional[int]
    optimize_n_best: Optional[int]
    optimize_rmsd_distinct_threshold: Optional[int]
    internuclear_distances: any
    leaderboard_size: int
    leaderboard_rmsd_threshold: Optional[float]
    rmsd_methods: List[str]
    alignment_methods: List[str]
    initial_optimization: bool
    max_evaluation_threads: Optional[int]
    max_optimization_threads: Optional[int]
    min_unoptimized_gradient_norm: Optional[float]
    initial_guess_path: Optional[str]
    equal_eval_opt: bool = False

    def __init__(
            self,
            max_optimization_steps: Optional[int],
            optimization_processes: Optional[int],
            optimization_threads: Optional[int],
            evaluator_processes: int,
            evaluator_threads: int,
            population_size: int,
            max_age: Optional[int],
            cluster_size: int,
            input_molecule: molecule,
            working_directory: str,
            checkpoint_frequency: Optional[int],
            optimize_every_nth_generation: Optional[int],
            optimize_n_best: Optional[int],
            optimize_rmsd_distinct_threshold: Optional[int],
            internuclear_distances: any,
            leaderboard_size: int,
            leaderboard_rmsd_threshold: Optional[float],
            alignment_methods: List[str],
            rmsd_methods: List[str],
            initial_optimization: bool,
            max_evaluation_threads: Optional[int],
            max_optimization_threads: Optional[int],
            min_unoptimized_gradient_norm: Optional[float],
            initial_guess_path: Optional[str]
    ):
        self.max_optimization_steps = max_optimization_steps
        self.optimization_processes = optimization_processes
        self.optimization_threads = optimization_threads
        self.evaluator_processes = evaluator_processes
        self.evaluator_threads = evaluator_threads
        self.population_size = population_size
        self.max_age = max_age
        self.cluster_size = cluster_size
        self.input_molecule = input_molecule
        self.working_directory = working_directory
        self.checkpoint_frequency = checkpoint_frequency
        self.optimize_every_nth_generation = optimize_every_nth_generation
        self.optimize_n_best = optimize_n_best
        self.optimize_rmsd_distinct_threshold = optimize_rmsd_distinct_threshold
        self.internuclear_distances = internuclear_distances
        self.leaderboard_size = leaderboard_size
        self.leaderboard_rmsd_threshold = leaderboard_rmsd_threshold
        self.alignment_methods = alignment_methods
        self.rmsd_methods = rmsd_methods
        self.initial_optimization = initial_optimization
        self.max_evaluation_threads = max_evaluation_threads
        self.max_optimization_threads = max_optimization_threads
        self.min_unoptimized_gradient_norm = min_unoptimized_gradient_norm
        self.initial_guess_path = initial_guess_path


class TimeStats:
    selection: float = 0
    crossover: float = 0
    mutation: float = 0
    optimization: float = 0
    evaluation: float = 0
    generator: float = 0
    terminating_conditions: float = 0
    invalid_molecules: float = 0
    leaderboard_management: float = 0

    variable_map = {
        'select': 'selection',
        'crossover': 'crossover',
        'mutate': 'mutation',
        'optimize': 'optimization',
        'evaluate': 'evaluation',
        'generate': 'generator',
        'terminate': 'terminating_conditions',
        'invalid': 'invalid_molecules',
        'leaderboard': 'leaderboard_management'
    }

    def __str__(self):
        stats: str = ''
        for key in self.variable_map.keys():
            variable = self.get_variable(key)
            stats += f'{key} {getattr(self, variable)}\n'
        return stats

    def get_variable(self, key: str):
        attribute = self.variable_map.get(key)
        if attribute:
            return attribute
        raise KeyError(f"Invalid key: {key}")

    def log_time(self, key: str, duration: float):
        variable_name = self.get_variable(key)
        setattr(self, variable_name, getattr(self, variable_name) + duration)

    def merge(self, other: 'TimeStats'):
        self.selection += other.selection
        self.crossover += other.crossover
        self.mutation += other.mutation
        self.optimization += other.optimization
        self.evaluation += other.evaluation
        self.terminating_conditions += other.terminating_conditions
        self.generator += other.generator
        self.invalid_molecules += other.invalid_molecules
        self.leaderboard_management += other.leaderboard_management

    def print(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Metric", "Time (seconds)"]
        for key in self.variable_map.keys():
            variable = self.get_variable(key)
            table.add_row([variable, f'{getattr(self, variable):0.5f}'])
        print(table)


class EvolutionEngine:
    optimizer: GeometryOptimizer
    evaluator: GeometryOptimizer
    generator: MoleculeGenerator
    selector: Selector
    crossover: Crossover
    mutators: [Mutator]
    terminators: [Terminator]
    elite: Optional[EliteSelector]
    params: EngineParameters

    def __init__(self, optimizer: GeometryOptimizer, generator: MoleculeGenerator, selector: Selector,
                 crossover: Crossover, mutators: [Mutator], terminators: [Terminator], params: EngineParameters,
                 elite: Optional[EliteSelector] = None, evaluator: GeometryOptimizer = None):
        if len(terminators) == 0:
            raise ValueError("Must have at least one terminator registered")
        self.optimizer = optimizer
        self.generator = generator
        self.evaluator = self.optimizer if evaluator is None else evaluator
        self.selector = selector
        self.crossover = crossover
        self.mutators = mutators
        self.elite = elite
        self.terminators = terminators
        self.params = params
        self.params.equal_eval_opt = evaluator.equals(self.optimizer)

    def set_params(self, params: EngineParameters):
        self.params = params

    def run(self) -> PopulationStats:
        print(self.info())
        time_stats: TimeStats = TimeStats()

        pop = load_initial_guesses(self.params.initial_guess_path, self.params.population_size,
                                   self.params.input_molecule, self.params.cluster_size,
                                   self.params.internuclear_distances)
        generated, gen_time = generate_population(population_size=self.params.population_size - len(pop),
                                            generator=self.generator,
                                            cluster_size=self.params.cluster_size,
                                            input_molecule=self.params.input_molecule,
                                            internuclear_distances=self.params.internuclear_distances)
        pop.extend(generated)
        time_stats.log_time('generate', gen_time)

        pop, eval_time = self.assign_fitness(pop)
        time_stats.log_time('evaluate', eval_time)

        if self.optimizer is not None and self.params.initial_optimization:
            pop, opt_time = self.optimize_geometries(pop)
            time_stats.log_time('optimize', opt_time)

            if not self.params.equal_eval_opt:
                pop, eval_time = self.assign_fitness(pop)
                time_stats.log_time('evaluate', eval_time)

        pop_stats: PopulationStats = PopulationStats(self.params.working_directory, self.params.rmsd_methods,
                                                     self.params.leaderboard_size,
                                                     self.params.leaderboard_rmsd_threshold)

        lb_time = pop_stats.add_all(get_best_individuals(pop, self.params.population_size))
        time_stats.log_time('leaderboard', lb_time)

        pop_stats.visualize()

        pop_stats = self.genetic_algorithm(pop, pop_stats, time_stats, 0)

        time_stats.print()
        self.__print_debug()

        return pop_stats

    def continue_from_checkpoint(self, checkpoint: RestoredCheckpoint):
        time_stats: TimeStats = checkpoint.time_stats
        pop: List[molecule] = checkpoint.current_pop
        pop_stats: PopulationStats = PopulationStats(self.params.working_directory, self.params.rmsd_methods,
                                                     self.params.leaderboard_size,
                                                     self.params.leaderboard_rmsd_threshold)
        self.__restore_state(checkpoint, pop_stats)

        pop_stats = self.genetic_algorithm(pop, pop_stats, time_stats, checkpoint.generation)

        time_stats.print()
        self.__print_debug()

        return pop_stats

    def genetic_algorithm(self, pop: List[molecule], pop_stats: PopulationStats,
                          time_stats: TimeStats, start_gen: int) -> PopulationStats:
        print('\n******************** GA START *********************')
        generation: int = start_gen
        end: bool = False
        while not end:
            generation += 1
            print('Applying genetic operators ...')
            pop, evo_stats = self.evolve_new_generation(pop, generation)
            time_stats.merge(evo_stats)
            print_fitness(pop, generation)

            lb_time = pop_stats.add_all(get_best_individuals(pop, self.params.population_size))
            time_stats.log_time('leaderboard', lb_time)

            task_start: float = time.perf_counter()
            for terminator in self.terminators:
                if terminator.check_condition(iteration=generation, new_best=pop_stats.get_best()):
                    print(f'\nGA terminated normally:\n  {terminator.get_reason()}\n')
                    end = True
                    break
            task_end: float = time.perf_counter()
            time_stats.log_time('terminate', task_end - task_start)

            if self.params.checkpoint_frequency is not None and generation % self.params.checkpoint_frequency == 0:
                self.__checkpoint(pop, pop_stats, time_stats, generation)

        print('\n******************** GA END *********************')
        return pop_stats

    def evolve_new_generation(self, pop: List[molecule], generation: int) -> tuple[list[molecule], TimeStats]:
        time_stats: TimeStats = TimeStats()

        pop_size: int = self.params.population_size
        new_generation: list[molecule] = []
        if self.elite is not None:
            new_generation.extend(self.elite.select(pop))

        for i in range(int((pop_size - self.elite.elite_quantity) / 2)):
            children, evo_stats = self.pairwise_evolution(pop)
            new_generation.extend(children)
            time_stats.merge(evo_stats)

        is_uneven: bool = ((pop_size - self.elite.elite_quantity) % 2) != 0
        if is_uneven:
            children, evo_stats = self.pairwise_evolution(pop)
            new_generation.append(children[0])
            time_stats.merge(evo_stats)

        new_generation, inv_time = self.handle_duplicate_molecules(new_generation)
        time_stats.log_time('invalid', inv_time)

        new_generation = self.age_population(new_generation)

        new_generation, inv_time = self.handle_invalid_molecules(new_generation)
        time_stats.log_time('invalid', inv_time)

        new_generation, eval_time = self.assign_fitness(new_generation)
        time_stats.log_time('evaluate', eval_time)

        if self.optimizer is not None and generation % self.params.optimize_every_nth_generation == 0:
            new_generation, opt_time = self.optimize_geometries(new_generation)
            time_stats.log_time('optimize', opt_time)

        return new_generation, time_stats

    def optimize_geometries(self, pop: List[molecule]) -> Tuple[List[molecule], float]:
        task_start: float = time.perf_counter()

        optimized_pop, unoptimized_pop = self.get_pop_for_optimization(pop)
        unoptimized_size: int = len(unoptimized_pop)

        if unoptimized_size == 0:
            task_end: float = time.perf_counter()
            return optimized_pop, task_end - task_start

        processes, threads = self.get_processes_and_threads(unoptimized_pop, 'optimization')
        if processes > 1:
            print(f'Starting parallel optimize_geometry ...')
            result: list[tuple[molecule, int]] = Parallel(n_jobs=processes)(
                delayed(self.optimize_geometry)(mol, threads[i % processes]) for i, mol in enumerate(unoptimized_pop)
            )
            self.optimizer.optimizations += unoptimized_size
            unoptimized_pop, error_flags = zip(*result)
            errors: int = sum(error_flags)
            self.optimizer.failed_optimizations += errors
            print(f'Successfully optimized the geometries of {unoptimized_size - errors}/{unoptimized_size} molecules ...')
            optimized_pop.extend(unoptimized_pop)
            task_end: float = time.perf_counter()
            return list(optimized_pop), task_end - task_start
        else:
            print(f'Starting sequential optimize_geometry ...')
            errors: int = 0
            for i in range(unoptimized_size):
                unoptimized_pop[i], error_flag = self.optimize_geometry(unoptimized_pop[i], threads[0])
                errors += error_flag
            self.optimizer.optimizations += unoptimized_size
            self.optimizer.failed_optimizations += errors
            print(f'Successfully optimized the geometries of {unoptimized_size - errors}/{unoptimized_size} molecules ...')
            optimized_pop.extend(unoptimized_pop)
            task_end: float = time.perf_counter()
            return optimized_pop, task_end - task_start

    def get_pop_for_optimization(self, pop: List[molecule]) -> Tuple[List[molecule], List[molecule]]:
        sorted_pop: List[molecule] = get_best_individuals(pop, self.params.population_size)
        optimized_pop: List[molecule] = []
        unoptimized_pop: List[molecule] = []
        current_pop_size: int = len(sorted_pop)

        optimize_best: bool = self.params.optimize_n_best is not None
        optimize_rmsd_distinct: bool = self.params.optimize_rmsd_distinct_threshold is not None
        optimize_min_gradient: bool = self.params.min_unoptimized_gradient_norm is not None

        while current_pop_size > 0:
            # If optimize_n_best is enabled
            if optimize_best and len(unoptimized_pop) >= self.params.optimize_n_best:
                for j in range(0, current_pop_size):
                    optimized_pop.append(pop[j])
                break

            current = sorted_pop[0]

            # If min_unoptimized_gradient_norm is enabled
            if optimize_min_gradient:
                grad_norm = current.get_gradient_norm()
                if grad_norm is not None and grad_norm < self.params.min_unoptimized_gradient_norm:
                    optimized_pop.append(current)
                    sorted_pop.remove(current)
                    current_pop_size -= 1
                    # i += 1
                    continue

            unoptimized_pop.append(current)

            # If optimize_rmsd_distinct_threshold is enabled
            if optimize_rmsd_distinct:
                rmsd_remove_molecules: List[molecule] = []
                for j in range(1, current_pop_size):
                    rmsd: float = sorted_pop[j].get_rmsd(current, self.params.rmsd_methods)
                    if rmsd <= self.params.optimize_rmsd_distinct_threshold:
                        rmsd_remove_molecules.append(sorted_pop[j])
                sorted_pop = [mol for mol in sorted_pop if mol not in rmsd_remove_molecules]
                optimized_pop.extend(rmsd_remove_molecules)
                current_pop_size = len(sorted_pop)

            sorted_pop.remove(current)
            current_pop_size -= 1

        print(f'Opt: pop_size: {len(pop)}, optimized size: {len(optimized_pop)}, unoptimized size: {len(unoptimized_pop)}')
        return optimized_pop, unoptimized_pop

    def get_processes_and_threads(self, pop: List[molecule], operation: str) -> Tuple[int, List[int]]:
        pop_size: int = len(pop)

        if pop_size == 0:
            return 0, []

        if operation == 'optimization' or (operation == 'evaluation' and self.evaluator is self.optimizer):
            if pop_size > self.params.optimization_processes:
                processes = self.params.optimization_processes
                threads = [self.params.optimization_threads for _ in range(processes)]
            else:
                max_threads: int = self.params.optimization_processes * self.params.optimization_threads
                processes = pop_size
                base_threads: int = max_threads // pop_size
                offset_threads: int = max_threads % pop_size
                threads = [base_threads + (1 if i < offset_threads else 0) for i in range(processes)]

            if self.params.max_optimization_threads is not None:
                for i in range(len(threads)):
                    if threads[i] > self.params.max_optimization_threads:
                        threads[i] = self.params.max_optimization_threads
        elif operation == 'evaluation':
            if pop_size > self.params.evaluator_processes:
                processes = self.params.evaluator_processes
                threads = [self.params.evaluator_threads for _ in range(processes)]
            else:
                max_threads: int = self.params.evaluator_processes * self.params.evaluator_threads
                processes = pop_size
                base_threads: int = max_threads // pop_size
                offset_threads: int = max_threads % pop_size
                threads = [base_threads + (1 if i < offset_threads else 0) for i in range(processes)]

            if self.params.max_evaluation_threads is not None:
                for i in range(len(threads)):
                    if threads[i] > self.params.max_evaluation_threads:
                        threads[i] = self.params.max_evaluation_threads
        else:
            raise RuntimeError(f'Invalid operation: {operation}')
        print(f'Number of {operation} processes: {processes}, threads: {threads}')
        return processes, threads

    def optimize_geometry(self, mol: molecule, threads: int) -> tuple[molecule, int]:
        max_steps = self.params.max_optimization_steps
        error_flag: int = 0
        optimized: Optional[molecule] = self.optimizer.optimize_geometry(mol, threads, max_steps)
        if optimized is not None:
            mol = optimized
        else:
            error_flag = 1
        return mol, error_flag

    def pairwise_evolution(self, pop: list[molecule]) -> tuple[list[molecule], TimeStats]:
        time_stats: TimeStats = TimeStats()

        task_start: float = time.perf_counter()
        parents = self.selector.select(pop, 2)
        child1, child2 = parents[0], parents[1]
        task_end: float = time.perf_counter()
        time_stats.log_time('select', task_end - task_start)

        task_start: float = time.perf_counter()
        child1, child2 = self.crossover.crossover(child1, child2)
        task_end: float = time.perf_counter()
        time_stats.log_time('crossover', task_end - task_start)

        task_start: float = time.perf_counter()
        for mutator in self.mutators:
            child1, child2 = mutator.mutate(child1), mutator.mutate(child2)
        task_end: float = time.perf_counter()
        time_stats.log_time('mutate', task_end - task_start)

        return [child1, child2], time_stats

    def handle_invalid_molecules(self, pop: list[molecule]) -> Tuple[List[molecule], float]:
        task_start: float = time.perf_counter()
        invalid: int = 0
        for i, mol in enumerate(pop):
            if not mol.has_fitness() and not mol.is_valid(self.params.internuclear_distances):
                pop[i] = self.generate_molecule()
                invalid += 1
        if invalid > 0:
            print(f'Regenerated {invalid} invalid molecules after genetic operators ...')
        task_end: float = time.perf_counter()
        return pop, task_end - task_start

    def handle_duplicate_molecules(self, pop: List[molecule]) -> Tuple[List[molecule], float]:
        task_start = time.perf_counter()

        seen_ids = set()
        new_pop: List[molecule] = []
        for mol in pop:
            mol_id = id(mol)
            if mol_id not in seen_ids:
                seen_ids.add(mol_id)
                new_pop.append(mol)
            else:
                new_pop.append(self.generate_molecule())

        task_end = time.perf_counter()
        return new_pop, task_end - task_start

    def generate_molecule(self) -> molecule:
        mol = self.generator.generate_molecule(input_molecule=self.params.input_molecule,
                                               cluster_size=self.params.cluster_size,
                                               internuclear_distances=self.params.internuclear_distances)
        return mol

    def age_population(self, pop: list[molecule]) -> list[molecule]:
        max_age = self.params.max_age
        if max_age is not None:
            for i, mol in enumerate(pop):
                mol.age += 1
                if mol.age >= max_age:
                    pop[i] = self.handle_old_molecule()
        return pop

    def handle_old_molecule(self) -> molecule:
        return self.generate_molecule()

    def assign_fitness(self, pop: list[molecule]) -> Tuple[List[molecule], float]:
        task_start: float = time.perf_counter()

        errors: int = 0
        evaluated_pop, unevaluated_pop = self.get_pop_for_evaluation(pop)
        unevaluated_size: int = len(unevaluated_pop)

        if unevaluated_size == 0:
            task_end: float = time.perf_counter()
            return evaluated_pop, task_end - task_start

        processes, threads = self.get_processes_and_threads(unevaluated_pop, 'evaluation')
        if processes > 1:
            print(f'Starting parallel assign_fitness ...')

            results = Parallel(n_jobs=processes)(
                delayed(self.calculate_fitness)(mol, threads[i % processes], i) for i, mol in enumerate(unevaluated_pop)
            )

            for index, mol, error_flag in results:
                unevaluated_pop[index] = mol
                errors += error_flag
        else:
            print(f'Starting sequential assign_fitness ...')
            for i in range(unevaluated_size):
                index, mol, error_flag = self.calculate_fitness(unevaluated_pop[i], threads[0], i)
                unevaluated_pop[index] = mol
                errors += error_flag

        self.evaluator.evals += unevaluated_size
        self.evaluator.failed_evals += errors
        if errors > 0:
            print(f'Regenerated {errors} molecules for which assign_fitness failed ...')

        evaluated_pop.extend(unevaluated_pop)
        task_end: float = time.perf_counter()
        return evaluated_pop, task_end - task_start

    def get_pop_for_evaluation(self, pop: List[molecule]) -> Tuple[List[molecule], List[molecule]]:
        evaluated_pop: List[molecule] = []
        unevaluated_pop: List[molecule] = []
        for mol in pop:
            if not self.params.equal_eval_opt or not mol.has_fitness():
                unevaluated_pop.append(mol)
            else:
                evaluated_pop.append(mol)
        print(f'Eval: pop_size: {len(pop)}, evaluated size: {len(evaluated_pop)}, unevaluated size: {len(unevaluated_pop)}')
        return evaluated_pop, unevaluated_pop

    def calculate_fitness(self, mol: molecule, threads: int, index: int) -> tuple[int, molecule, int]:
        errors: int = 0
        calculated: Optional[molecule] = self.evaluator.calculate_energy(mol, threads)
        if calculated is None or not calculated.has_fitness():
            while calculated is None or not calculated.has_fitness():
                errors += 1
                if errors > 10:
                    raise RuntimeError(f'Unable to calculate energy of a molecule:\n{mol.get_xyz_string()}')

                calculated = self.generate_molecule()
                calculated = self.evaluator.calculate_energy(calculated, threads)
        mol = calculated
        return index, mol, errors

    def __print_debug(self) -> None:
        info: str = ''
        for mutator in self.mutators:
            debug: Optional[str] = mutator.debug_str()
            if debug is not None:
                info += debug + '\n'
        crossover_debug: Optional[str] = self.crossover.debug_str()
        if crossover_debug is not None:
            info += crossover_debug + '\n'
        info += f'Evaluator performed {self.evaluator.evals} evaluations. ({self.evaluator.failed_evals} failed)\n'
        if self.optimizer is not None:
            info += (f'Optimizer performed {self.optimizer.optimizations} optimizations. '
                     f'({self.optimizer.failed_optimizations} failed)\n')
        info += f'Generator missed {self.generator.missed_runs} runs.\n'
        print('DEBUG INFO:\n' + info)

    def __checkpoint(self, pop: List[molecule], pop_stats: PopulationStats, time_stats: TimeStats,
                     generation: int) -> None:
        save_checkpoint(self.params.working_directory, pop, pop_stats, time_stats, generation, self.crossover,
                        self.terminators, self.mutators, self.evaluator, self.optimizer)

    def info(self) -> str:
        width: int = 12
        info: str = ''
        info += '*' * width + ' Genetic Algorithm Run Info ' + '*' * width + '\n'
        header: int = len(info) - 1
        info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Population Info ' + '*' * int(width / 4) + '\n'
        info += (' ' * int(width / 3) + f' Element: {self.params.input_molecule.atoms[0].element_symbol}\n' +
                 ' ' * int(width / 3) + f' Charge: {self.params.input_molecule.charge}\n' +
                 ' ' * int(width / 3) + f' Multiplicity: {self.params.input_molecule.multiplicity}\n' +
                 ' ' * int(width / 3) + f' Cluster size: {self.params.cluster_size}\n' +
                 ' ' * int(width / 3) + f' Population size: {self.params.population_size}\n' +
                 ' ' * int(width / 3) + f' Max age: {self.params.max_age}\n' +
                 ' ' * int(width / 3) + f' RMSD methods: {self.params.rmsd_methods}\n' +
                 ' ' * int(width / 3) + f' Alignment methods: {self.params.alignment_methods}\n' +
                 ' ' * int(width / 3) + f' Generator: {self.generator}\n')

        info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Results Config ' + '*' * int(width / 4) + '\n'
        info += (' ' * int(width / 3) + f' Leaderboard size: {self.params.leaderboard_size}\n' +
                 ' ' * int(width / 3) + f' Leaderboard RMSD threshold: {self.params.leaderboard_rmsd_threshold}\n' +
                 ' ' * int(width / 3) + f' Checkpoint frequency: {self.params.checkpoint_frequency}\n' +
                 ' ' * int(width / 3) + f' Working directory: {self.params.working_directory}\n')

        info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Operators Info ' + '*' * int(width / 4) + '\n'
        info += (' ' * int(width / 3) + f' Selector:  {self.selector}\n' +
                 ' ' * int(width / 3) + f' Crossover: {self.crossover}\n' +
                 ' ' * int(width / 3) + f' Mutators:\n')
        for mutator in self.mutators:
            info += ' ' * int(width / 3) + f'  {mutator}\n'
        if self.elite is not None:
            info += ' ' * int(width / 3) + f' Elites:    {self.elite}\n'
        info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Terminating conditions ' + '*' * int(width / 4) + '\n'
        for i, terminator in enumerate(self.terminators):
            info += ' ' * int(width / 3) + f' {i}. {terminator}\n'
        if self.optimizer is not None:
            info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Optimizer config ' + '*' * int(width / 4) + '\n'
            info += (' ' * int(width / 3) + f' Program: {self.optimizer.method_program}\n' +
                     ' ' * int(width / 3) + f' Method: {self.optimizer.method}\n' +
                     ' ' * int(width / 3) + f' Optimization program: {self.optimizer.optimization_program}\n' +
                     ' ' * int(width / 3) + f' Processes: {self.params.optimization_processes}\n' +
                     ' ' * int(width / 3) + f' Threads: {self.params.optimization_threads}\n' +
                     ' ' * int(width / 3) + f' Max Geometry steps: {self.params.max_optimization_steps}\n' +
                     ' ' * int(width / 3) + f' Initial optimization: {self.params.initial_optimization}\n')
            if self.params.optimize_every_nth_generation is not None:
                info += ' ' * int(width / 3) + f' Optimize every n-th generation: {self.params.optimize_every_nth_generation}\n'
            if self.params.optimize_n_best is not None:
                info += ' ' * int(width / 3) + f' Optimize n best: {self.params.optimize_n_best}\n'
            if self.params.optimize_rmsd_distinct_threshold is not None:
                info += ' ' * int(width / 3) + f' Optimize distinct rmsd threshold: {self.params.optimize_rmsd_distinct_threshold}\n'
        info += ' ' * int(width / 4) + '*' * int(width / 4) + ' Evaluator config ' + '*' * int(width / 4) + '\n'
        if self.evaluator is not self.optimizer:
            info += (' ' * int(width / 3) + f' Program: {self.evaluator.method_program}\n' +
                     ' ' * int(width / 3) + f' Method: {self.evaluator.method}\n')
            info += (' ' * int(width / 3) + f' Processes: {self.params.evaluator_processes}\n' +
                     ' ' * int(width / 3) + f' Threads: {self.params.evaluator_threads}\n')
        else:
            info += ' ' * int(width / 3) + f' <Using optimizer for evaluations>\n'
        info += '*' * header
        return info

    def __restore_state(self, checkpoint: RestoredCheckpoint, pop_stats: PopulationStats) -> None:
        for i in range(len(checkpoint.leaderboard)):
            pop_stats.leaderboard[i] = checkpoint.leaderboard[i]

        for mutator in self.mutators:
            restored_stats = checkpoint.operator_stats.get(mutator.get_name())
            if restored_stats is not None:
                mutator.update_stats(restored_stats)
        for terminator in self.terminators:
            restored_values = checkpoint.terminator_stats.get(terminator.get_name())
            if restored_values is not None:
                terminator.update_terminator_values(restored_values)
        crossover_stats = checkpoint.operator_stats.get(self.crossover.get_name())
        if crossover_stats is not None:
            self.crossover.update_stats(crossover_stats)

        self.evaluator.update_eval_stats(checkpoint.evaluator_stats)
        if checkpoint.optimizer_stats is not None:
            self.optimizer.update_optimization_stats(checkpoint.optimizer_stats)
