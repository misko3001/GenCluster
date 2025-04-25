import os
from typing import List, Optional, Dict, Tuple

import yaml
from pydantic import BaseModel, field_validator, Field, model_validator, ValidationError

from optimization.genetic.molecule.MoleculeUtils import AlignmentMethods, RMSDMethods
from optimization.genetic.operations.Terminator import DurationTerminator
from utils.ClassUtils import get_class_names
from utils.FileUtils import is_absolute_path

os.environ['PYDANTIC_ERRORS_INCLUDE_URL'] = '0'
mutator_file_path = 'optimization/genetic/operations/Mutator.py'
terminator_file_path = 'optimization/genetic/operations/Terminator.py'
crossover_file_path = 'optimization/genetic/operations/Crossover.py'
selector_file_path = 'optimization/genetic/operations/Selector.py'
generator_file_path = 'optimization/genetic/molecule/MoleculeGenerator.py'
mutator_names = []
terminator_names = []
crossover_names = []
selector_names = []
generator_names = []


class InvalidConfiguration(Exception):
    pass


class ProblemConfig(BaseModel):
    """
    Represents the configuration for a problem setup in the system.

    This class defines the required and optional configuration parameters for
    setting up a problem. The configuration includes details like the
    input file path for the molecule, size of the cluster, directory for working
    files, and checkpoint frequency.

    :ivar input_molecule_path: The file path to the molecule input file. [Required]
    :type input_molecule_path: str
    :ivar cluster_size: Size of the molecule cluster defined for the problem. [Required]
    :type cluster_size: int
    :ivar molecule_charge: Charge of the input molecule. [Required]
    :type molecule_charge: float
    :ivar molecule_multiplicity: Multiplicity of the input molecule. [Required]
    :type molecule_multiplicity: float
    :ivar working_directory: The path to the working directory for files. Defaults to 'data/out'. [Optional]
    :type working_directory: Optional[str]
    :ivar checkpoint_every_nth_generation: Frequency (every nth generation) for
        checkpoint creation, if not provided, checkpoints will not be created. [Optional]
    :type checkpoint_every_nth_generation: Optional[int]
    :ivar internuclear_angstrom_distances: Defines internuclear distances between every atom in a cluster [Required]
    :type internuclear_angstrom_distances: Dict[str, Tuple[float, float]]
    :ivar leaderboard_size: Defines the maximum number of the best molecules returned after evolution. [Required]
    :type leaderboard_size: int
    :ivar leaderboard_rmsd_threshold: Defines the minimum RMSD for replacing other molecules on the leaderboard. [Optional]
    :type leaderboard_rmsd_threshold: Optional[float]
    """
    input_molecule_path: str
    cluster_size: int
    molecule_charge: int = Field(ge=0)
    molecule_multiplicity: int = Field(ge=0)
    working_directory: Optional[str] = None
    checkpoint_every_nth_generation: Optional[int] = None
    internuclear_angstrom_distances: Dict[str, Tuple[float, float]]
    leaderboard_size: int = Field(ge=1)
    leaderboard_rmsd_threshold: Optional[float] = None
    rmsd_methods: Optional[List[str]] = None
    alignment_methods: Optional[List[str]] = None

    @field_validator('input_molecule_path', mode='after')
    def check_input_molecule_path_exists(cls, value: str) -> str:
        if is_absolute_path(value) and not os.path.isfile(value):
            raise ValueError(f'input_molecule_path does not point to a valid file')
        return value

    @field_validator('working_directory', mode='after')
    def check_optional_working_directory(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not os.path.isdir(value):
            raise ValueError(f'working_directory does not point to a valid directory')
        return value

    @field_validator('leaderboard_rmsd_threshold', mode='after')
    def check_optional_leaderboard_rmsd_threshold(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError(f'leaderboard_rmsd_threshold must be greater than 0')
        return value

    @field_validator('checkpoint_every_nth_generation', mode='after')
    def check_optional_checkpoint_every_nth_generation(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f'checkpoint_every_nth_generation must be greater than 0')
        return value

    @field_validator('internuclear_angstrom_distances', mode='after')
    def check_internuclear_angstrom_distances(cls, value: Dict[str, Tuple[float, float]]) -> Dict[
        str, Tuple[float, float]]:
        for distance in value.values():
            min_angstroms, max_angstroms = distance
            if min_angstroms <= 0 or max_angstroms <= 0:
                raise ValueError(f'internuclear_angstrom_distances must be greater than 0')
            if min_angstroms > max_angstroms:
                raise ValueError(f'minimum distances must be greater than or equal to maximum distances' +
                                 f' (provided min: {min_angstroms}, max: {max_angstroms})')

        return value

    @field_validator('alignment_methods', mode='after')
    def check_optional_alignment_methods(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None:
            for method in value:
                if method not in AlignmentMethods:
                    raise ValueError(f'Unknown alignment method: {method}')
        return value

    @field_validator('rmsd_methods', mode='after')
    def check_optional_rmsd_methods(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None:
            for method in value:
                if method not in RMSDMethods:
                    raise ValueError(f'Unknown RMSD method: {method}')
        return value

    @field_validator("internuclear_angstrom_distances", mode='before')
    def parse_distances(cls, value: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        parsed = {}
        for key, val in value.items():
            if isinstance(val, str):
                try:
                    val = tuple(map(float, val.split(",")))
                except ValueError:
                    raise ValueError(f"Invalid tuple format for {key}: {val}")
            parsed[key] = val
        return parsed


class PopulationConfig(BaseModel):
    """
    Represents a configuration for managing a population, including size and optional
    attributes such as maximum age.

    :ivar size: The size of the population. Must be greater than or equal to 2. [Required]
    :type size: int
    :ivar max_age: The maximum age of members in the population.
                   If defined, it must be greater than or equal to 1. [Optional]
    :type max_age: Optional[int]
    :ivar initial_guess_path: Path to the initial guesses file for initial population [Optional]
    :type initial_guess_path: Optional[str]
    """
    size: int = Field(ge=2)
    max_age: Optional[int] = None
    initial_guess_path: Optional[str] = None

    @field_validator('max_age', mode='after')
    def check_optional_max_age(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f'optimize_every_nth_generation must be greater than 0')
        return value

    @field_validator('initial_guess_path', mode='after')
    def check_optional_initial_guess_path(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and is_absolute_path(value) and not os.path.isfile(value):
            raise ValueError(f'initial_guess_path does not point to a valid file')
        return value


class ModelConfig(BaseModel):
    """
    Encapsulates configuration details for a model. This model will be either used to optimize molecule geometries
    or evaluate their energies.

    :ivar program: Name of the program that will be used to perform the optimization or evaluation. [Required]
    :type program: str
    :ivar method: Algorithm to be used for calculation in requested program. [Required]
    :type method: str
    :ivar threads: Number of threads to be used for each calculation. [Required]
    :type threads: int
    :ivar config_path: Optional value that sets the maximum number of threads to be used in a single process [Optional]
    :type config_path: Optional[int]
    :ivar config_path: Optional path to a configuration file. If provided, it must point to a
        valid file path on the filesystem. [Optional]
    :type config_path: Optional[str]
    """
    program: str
    method: str
    threads: int = Field(ge=1)
    max_threads: Optional[int] = None
    config_path: Optional[str] = None

    @field_validator('config_path', mode='after')
    def check_optional_config_path(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and is_absolute_path(value) and not os.path.isfile(value):
            raise ValueError(f'config_path does not point to a valid file')
        return value

    @field_validator('max_threads', mode='after')
    def check_optional_max_threads(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f'max_threads must be greater than or equal to 1')
        return value


class OptimizerArgs(BaseModel):
    """
    Represents arguments for modifying the optimization process.

    This class is used to define parameters controlling the behavior of
    optimization tasks. It specifies configuration values like how frequently
    optimization occurs, the number of best results to optimize, and thresholds
    for RMSD (Root Mean Square Deviation) distinctiveness.

    :ivar optimize_every_nth_generation: Specifies how many generations must pass
        between optimization occurrences. If not set defaults to 1. [Optional]
    :type optimize_every_nth_generation: Optional[int]
    :ivar optimize_n_best: Defines the number of top-performing molecules to include
        in optimization. If not set, all molecules are optimized. [Optional]
    :type optimize_n_best: Optional[int]
    :ivar optimize_rmsd_distinct_threshold: Determines the minimum RMSD distinctiveness
        required for optimization. If not set, no threshold is imposed. [Optional]
    :type optimize_rmsd_distinct_threshold: Optional[float]
    :ivar min_unoptimized_gradient_norm: Determines if a molecule will be optimized based on if it has greater gradient
        morm than the minimum gradient norm (if available). If not set, no constraint is imposed. [Optional]
    :type min_unoptimized_gradient_norm: Optional[float]
    """
    optimize_every_nth_generation: Optional[int] = None
    optimize_n_best: Optional[int] = None
    optimize_rmsd_distinct_threshold: Optional[float] = None
    min_unoptimized_gradient_norm: Optional[float] = None

    @field_validator('optimize_every_nth_generation', mode='after')
    def check_optional_optimize_every_nth_generation(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f'optimize_every_nth_generation must be greater than 0')
        return value

    @field_validator('optimize_n_best', mode='after')
    def check_optional_optimize_n_best(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f'optimize_n_best must be greater than 0')
        return value

    @field_validator('optimize_rmsd_distinct_threshold', mode='after')
    def check_optional_optimize_rmsd_distinct_threshold(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError(f'optimize_rmsd_distinct_threshold must be greater than or equal to 0')
        return value

    @field_validator('min_unoptimized_gradient_norm', mode='after')
    def check_optional_min_unoptimized_gradient_norm(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError(f'min_unoptimized_gradient_norm must be greater than or equal to 0')
        return value


class OptimizerConfig(BaseModel):
    """
    Represents the configuration required for the optimization process.

    This class defines the parameters and constraints needed for executing
    optimization tasks. It integrates necessary details like the optimization
    program, the algorithm to be used, number of processes, as well as the
    maximum optimization steps allowed. It also encapsulates model configurations
    and optional arguments to tailor the optimization process.

    :ivar optimization_program: The name of the optimization program to be used.
    :type optimization_program: str
    :ivar XTB_optimization_level: Sets the level of XTB optimization. Can be crude, sloppy, loose, lax,
        normal, tight, vtight or extreme. Defaults to extreme. [Optional].
    :type XTB_optimization_level: Optional[str]
    :ivar ASE_optimization_algorithm: The algorithm for ASE optimization. If not set defaults to 'LBFGS'. [Optional]
    :type ASE_optimization_algorithm: Optional[str]
    :ivar processes: The number of processes to utilize for
        the optimization, must be greater than or equal to 1. [Required]
    :type processes: int
    :ivar max_optimization_steps: Maximum number of steps allowed
        in the optimization process, must be greater than or equal to 1. [Optional - if using Native program]
    :ivar ASE_forces_convergence_criterion: If using ASE specifies the convergence criterion in eV/Angstroms.
        If not set will be set to the default ASE value 0.02 eV/Angstroms [Optional]
    :type ASE_forces_convergence_criterion: Optional[float]
    :ivar initial_optimization: If set to True, optimization will be performed after generating population. [Required]
    :type initial_optimization: bool
    :ivar debug: If set to true, will save debug files in working directory. [Required]
    :type debug: bool
    :ivar model: The configuration of the model which for example specifies the program to be used. [Required]
    :type model: ModelConfig
    :ivar args: Optional additional arguments for the optimizer. [Optional]
    :type args: Optional[OptimizerArgs]
    """
    optimization_program: str = Field(min_length=1)
    XTB_optimization_level: Optional[str] = None
    ASE_optimization_algorithm: Optional[str] = None
    processes: int = Field(ge=1)
    max_optimization_steps: Optional[int] = None
    ASE_forces_convergence_criterion: Optional[float] = None
    initial_optimization: bool
    debug: bool
    model: ModelConfig
    args: Optional[OptimizerArgs] = None

    @field_validator('ASE_optimization_algorithm', mode='after')
    def check_optional_ase_optimization_algorithm(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError(f'ASE_optimization_algorithm cannot be empty')
        return value

    @field_validator('XTB_optimization_level', mode='after')
    def check_optional_xtb_optimization_level(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError(f'XTB_optimization_level cannot be empty')
        elif value not in ['crude', 'sloppy', 'loose', 'lax', 'normal', 'tight', 'vtight', 'extreme']:
            raise ValueError(f'Unknown XTB optimization level: {value}')
        return value

    @field_validator('ASE_forces_convergence_criterion', mode='after')
    def check_optional_ase_forces_convergence_criterion(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError(f'ASE_forces_convergence_criterion must be greater than 0')
        return value

    @field_validator('max_optimization_steps', mode='after')
    def check_optional_max_optimization_steps(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f'max_optimization_steps must be greater than 0')
        return value


class EvaluatorConfig(BaseModel):
    """
    Represents configuration for an evaluator in the application.

    This class defines the configuration necessary for an evaluator to function
    appropriately. It includes information such as the number of processes for
    parallelization as well as the configuration for a model which will be used
    to evaluate the molecules.

    :ivar processes: Specifies the number of processes to be used for
        parallel evaluations, if set to 1 the evaluation will be run in serial mode. [Required]
    :type processes: int
    :ivar debug: If set to true, will save debug files in working directory. [Required]
    :type debug: bool
    :ivar model: Holds the configuration details of the model that the evaluator uses to evaluate molecules. [Required]
    :type model: ModelConfig
    """
    processes: int = Field(ge=1)
    debug: bool
    model: ModelConfig


class GeneratorArgs(BaseModel):
    """
    Represents the arguments required for generating molecules.

    :ivar min_angstrom_distance: The minimum allowed angstrom distance for generating. [Required]
    :type min_angstrom_distance: float
    :ivar max_angstrom_distance: The maximum allowed angstrom distance for generating. [Required]
    :type max_angstrom_distance: float
    """
    min_angstrom_distance: float = Field(gt=0)
    max_angstrom_distance: float = Field(gt=0)


class GeneratorConfig(BaseModel):
    """
        Configuration class for a generator.

        This class defines the configuration required for a generator.

        :ivar name: Name of the generator. [Required]
            (Available generator names: PackMolGenerator, RandomGeometryGenerator)
        :type name: str
        """
    name: str
    args: GeneratorArgs

    @field_validator('name', mode='after')
    def check_selector_name(cls, value: str) -> str:
        if value not in generator_names:
            raise ValueError(f'Unrecognized generator name: "{value}" (Available generator names: {generator_names})')
        return value


class SelectorArgs(BaseModel):
    """
    Represents the arguments for a requested selector.

    :ivar tournament_size: Required and used only in (Rmsd)TournamentSelector. Determines the tournament size for
        each selection. [Optional]
    :type tournament_size: int
    """
    tournament_size: Optional[int] = None

    @field_validator('tournament_size', mode='after')
    def check_optional_tournament_size(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 2:
            raise ValueError(f'tournament_size must be greater than 1')
        return value


class SelectorConfig(BaseModel):
    """
    Configuration class for a selector.

    This class defines the configuration required for a selector. It includes the
    name of the selector, optional number of elites and its corresponding arguments.

    :ivar name: Name of the selector. [Required]
        (Available selector names: TournamentSelector, RmsdTournamentSelector)
    :type name: str
    :ivar elite_quantity: Number of elite individuals that should be passed on without modification. [Optional]
    :type elite_quantity: Optional[int]
    :ivar args: Arguments required for the selector, encapsulated in a `SelectorArgs` object. [Required]
    :type args: SelectorArgs
    """
    name: str
    elite_quantity: Optional[int] = None
    args: SelectorArgs

    @field_validator('name', mode='after')
    def check_selector_name(cls, value: str) -> str:
        if value not in selector_names:
            raise ValueError(f'Unrecognized selector name: "{value}" (Available selector names: {selector_names})')
        return value

    @field_validator('elite_quantity', mode='after')
    def check_optional_elite_quantity(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError(f'elite_quantity cannot be a negative number')
        return value


class CrossoverArgs(BaseModel):
    """
    Represents the arguments for a requested crossover operator.

    :ivar probability: The probability of applying crossover. [Required]
    :type probability: float
        """
    probability: float = Field(ge=0, le=1)


class CrossoverConfig(BaseModel):
    """
    Configuration class for a crossover.

    This class defines the configuration required for a crossover. It includes the
    name of the crossover and its corresponding arguments.

    :ivar name: Name of the crossover. [Required]
        (Available crossover names: TwoPointCrossover)
    :type name: str
    :ivar args: Arguments required for the crossover, encapsulated in a `CrossoverArgs` object. [Required]
    :type args: CrossoverArgs
    """
    name: str
    args: CrossoverArgs

    @field_validator('name', mode='after')
    def check_crossover_name(cls, value: str) -> str:
        if value not in crossover_names:
            raise ValueError(f'Unrecognized crossover name: "{value}" (Available crossover names: {crossover_names})')
        return value


class TerminatorArgs(BaseModel):
    """
    Represents the arguments for a termination condition in a requested terminator.

    :ivar max_iterations: Required and used only in MaxIterationsTerminator.
        Maximum number of iterations (generations), after which algorithm is stopped. [Optional]
    :type max_iterations: Optional[int]
    :ivar max_stagnating_iterations: Required and used only in StagnationTerminator.
        Maximum number of iterations allowed without improvement. [Optional]
    :type max_stagnating_iterations: Optional[int]
    :ivar unit_type: Required and used only in DurationTerminator.
        Time unit type for the time limit. [Optional]
    :type unit_type: Optional[str]
    :ivar unit_value: Required and used only in DurationTerminator.
        Unit value for the time limit (combines with `unit_type`). [Optional]
    :type unit_value: Optional[float]
    :ivar last_m_generations: Required and used only in ConvergenceTerminator.
        Specifies the maximum number of generations on which convergence should be checked. [Optional]
    :type last_m_generations: Optional[float]
    :ivar last_n_generations: Required and used only in ConvergenceTerminator.
        Specifies the subset of `last_m_generations` based on which delta will be calculated. [Optional]
    :type last_n_generations: Optional[float]
    :ivar delta: Required and used only in ConvergenceTerminator.
        The threshold for convergence. It specifies the maximum allowed difference between the average fitness
        of the last `m` generations and the average fitness of the last `n` generations. [Optional]
    :type delta: Optional[float]
    """
    max_iterations: Optional[int] = None
    max_stagnating_iterations: Optional[int] = None
    unit_type: Optional[str] = None
    unit_value: Optional[float] = None
    last_m_generations: Optional[int] = None
    last_n_generations: Optional[int] = None
    delta: Optional[float] = None

    @field_validator('max_iterations', mode='after')
    def check_optional_max_iterations(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f'max_iterations must be greater than 0')
        return value

    @field_validator('max_stagnating_iterations', mode='after')
    def check_optional_max_stagnating_iterations(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f'max_stagnating_iterations must be greater than 0')
        return value

    @field_validator('unit_type', mode='after')
    def check_optional_unit_type(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value.casefold() not in DurationTerminator.TIME_UNITS:
            raise ValueError(f'unit_type must be one of supported units: {DurationTerminator.TIME_UNITS}')
        return value

    @field_validator('unit_value', mode='after')
    def check_optional_unit_value(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError(f'unit_value must be greater than 0')
        return value

    @field_validator('last_m_generations', mode='after')
    def check_optional_last_m_generations(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f'last_m_generations must be greater than 0')
        return value

    @field_validator('last_n_generations', mode='after')
    def check_optional_last_n_generations(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f'last_n_generations must be greater than 0')
        return value

    @field_validator('delta', mode='after')
    def check_optional_delta(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError(f'delta must be greater than 0')
        return value


class TerminatorConfig(BaseModel):
    """
    Configuration class for a terminator.

    This class defines the configuration required for a terminator. It includes the
    name of the terminator and its corresponding arguments.

    :ivar name: Name of the terminator. [Required]
        (Available terminator names: MaxIterationsTerminator, StagnationTerminator)
    :type name: str
    :ivar args: Arguments required for the mutator, encapsulated in a `TerminatorArgs` object. [Required]
    :type args: TerminatorArgs
    """
    name: str
    args: TerminatorArgs

    @model_validator(mode='after')
    def check_terminator_config(self):
        if self.name == 'MaxIterationsTerminator':
            TerminatorConfig.check_max_iterations_terminator_args(self.args)
        elif self.name == 'StagnationTerminator':
            TerminatorConfig.check_stagnation_terminator_args(self.args)
        elif self.name == 'DurationTerminator':
            TerminatorConfig.check_duration_terminator_args(self.args)
        return self

    @field_validator('name', mode='after')
    def check_terminator_name(cls, value: str) -> str:
        if value not in terminator_names:
            raise ValueError(
                f'Unrecognized terminator name: "{value}" (Available terminator names: {terminator_names})')
        return value

    @staticmethod
    def check_stagnation_terminator_args(args: TerminatorArgs) -> None:
        if args.max_stagnating_iterations is None:
            raise ValueError('max_stagnating_iterations must be specified for StagnationTerminator')

    @staticmethod
    def check_max_iterations_terminator_args(args: TerminatorArgs) -> None:
        if args.max_iterations is None:
            raise ValueError('max_iterations must be specified for MaxIterationsTerminator')

    @staticmethod
    def check_duration_terminator_args(args: TerminatorArgs) -> None:
        if args.unit_type is None:
            raise ValueError('unit_type must be specified for DurationTerminator')
        if args.unit_value is None:
            raise ValueError('unit_value must be specified for DurationTerminator')

    @staticmethod
    def check_convergence_terminator_args(args: TerminatorArgs) -> None:
        if args.last_m_generations is None:
            raise ValueError('last_m_generations must be specified for ConvergenceTerminator')
        if args.last_n_generations is None:
            raise ValueError('last_n_generations must be specified for ConvergenceTerminator')
        if args.delta is None:
            raise ValueError('delta must be specified for ConvergenceTerminator')
        if args.last_m_generations <= args.last_n_generations:
            raise ValueError('last_m_generations must be greater than last_n_generations for ConvergenceTerminator')


class MutatorArgs(BaseModel):
    """
    Represents the arguments for a requested mutator.

    This class defines various parameters that control the behavior of mutation operations.

    :ivar probability: The probability of applying a mutation. [Required]
    :type probability: float
    :ivar per_atom_probability: Required and used only for CreepMutator. The probability of creep mutation being applied
        to each individual atom. [Optional]
    :type per_atom_probability: Optional[float]
    :ivar max_mutation_strength: Required and used only for CreepMutator. The maximum strength of mutation,
     if specified, where the value sets the maximum displacement in a single dimension. [Optional]
    :type max_mutation_strength: Optional[float]
    """
    probability: float = Field(ge=0, le=1)
    per_atom_probability: Optional[float] = None
    max_mutation_strength: Optional[float] = None

    @field_validator('per_atom_probability', mode='after')
    def check_optional_per_atom_probability(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if value <= 0:
                raise ValueError(f'per_atom_probability must be greater than 0')
            if value > 1:
                raise ValueError(f'per_atom_probability must be less than or equal to 1')
        return value

    @field_validator('max_mutation_strength', mode='after')
    def check_optional_max_mutation_strength(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError(f'max_mutation_strength must be greater than 0')
        return value


class MutatorConfig(BaseModel):
    """
    Configuration class for a mutator.

    This class defines the configuration required for a mutator. It includes the
    name of the mutator and its corresponding arguments.

    :ivar name: Name of the mutator. [Required]
        (Available mutator names: CreepMutator, TwinningMutator)
    :type name: str
    :ivar args: Arguments required for the mutator, encapsulated in a `MutatorArgs` object. [Required]
    :type args: MutatorArgs
    """
    name: str
    args: MutatorArgs

    @model_validator(mode='after')
    def check_mutator_config(self):
        if self.name == 'CreepMutator':
            MutatorConfig.check_creep_mutator_args(self.args)
        return self

    @field_validator('name', mode='after')
    def check_mutator_name(cls, value: str) -> str:
        if value not in mutator_names:
            raise ValueError(f'Unrecognized mutator name: "{value}" (Available mutator names: {mutator_names})')
        return value

    @staticmethod
    def check_creep_mutator_args(args: MutatorArgs) -> None:
        if args.per_atom_probability is None:
            raise ValueError('per_atom_probability must be specified for CreepMutator')
        if args.max_mutation_strength is None:
            raise ValueError('max_mutation_strength must be specified for CreepMutator')


class AppConfig(BaseModel):
    """
    Represents the configuration settings for an application including population,
    problem definitions, genetic algorithm parameters, and their validation.

    This class serves as a container for all main configuration components
    necessary to set up and run a genetic algorithm. It validates
    the presence of required configurations, such as terminators, mutators, and an
    evaluator or optimizer.

    :ivar mutators: A list of dictionaries, each holding the mutator name along
                    with its configuration.
    :type mutators: List[Dict[str, MutatorConfig]]
    :ivar terminators: A list of dictionaries, each holding the terminator name
                       along with its configuration.
    :type terminators: List[Dict[str, TerminatorConfig]]
    :ivar crossover: Configuration settings for the crossover operator.
    :type crossover: CrossoverConfig
    :ivar selector: Configuration settings for the selection operator. Defines the
                    method for selecting individuals.
    :type selector: SelectorConfig
    :ivar generator: Configuration settings for the generator, which creates the
                     initial population of solutions.
    :type generator: GeneratorConfig
    :ivar evaluator: Configuration settings for the evaluator. Can be None if optimizer is defined.
    :type evaluator: Optional[EvaluatorConfig]
    :ivar optimizer: Configuration settings for the optimizer. Can be None if optimization is not required.
    :type optimizer: Optional[OptimizerConfig]
    :ivar population: Configuration settings for the population.
    :type population: PopulationConfig
    :ivar problem: Configuration settings for the problem being solved.
    :type problem: ProblemConfig
    """
    mutators: List[Dict[str, MutatorConfig]]
    terminators: List[Dict[str, TerminatorConfig]]
    crossover: CrossoverConfig
    selector: SelectorConfig
    generator: GeneratorConfig
    evaluator: Optional[EvaluatorConfig] = None
    optimizer: Optional[OptimizerConfig] = None
    population: PopulationConfig
    problem: ProblemConfig

    @model_validator(mode='after')
    def check_app_config(self):
        if len(self.terminators) == 0:
            raise ValueError('No terminators have been provided')
        if len(self.mutators) == 0:
            print("\033[93mWarning: No mutators have been provided\033[0m")
        if self.optimizer is None and self.evaluator is None:
            raise ValueError('No evaluator has been provided')
        if self.optimizer is not None:
            if self.optimizer.max_optimization_steps is None and self.optimizer.optimization_program.casefold() != 'native':
                raise ValueError(f'max_optimization_steps is not set for non-native optimization program.')
        elites = self.selector.elite_quantity
        if elites is not None and elites > self.population.size:
            raise ValueError('Elite quantity cannot be greater than population size')


def load_app_config(yaml_path: str, project_dir: Optional[str] = None) -> AppConfig:
    update_name_file_paths(project_dir)
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    try:
        return AppConfig(**yaml_data)
    except ValidationError as e:
        raise InvalidConfiguration(e) from None

def update_name_file_paths(project_dir: Optional[str]):
    if project_dir is not None:
        global mutator_file_path, terminator_file_path, crossover_file_path, selector_file_path, generator_file_path
        mutator_file_path = project_dir + '/' + mutator_file_path
        terminator_file_path = project_dir + '/' + terminator_file_path
        crossover_file_path = project_dir + '/' + crossover_file_path
        selector_file_path = project_dir + '/' + selector_file_path
        generator_file_path = project_dir + '/' + generator_file_path
    global mutator_names, terminator_names, crossover_names, selector_names, generator_names
    mutator_names = get_class_names(mutator_file_path, ['Mutator'])
    terminator_names = get_class_names(terminator_file_path, ['Terminator'])
    crossover_names = get_class_names(crossover_file_path, ['Crossover'])
    selector_names = get_class_names(selector_file_path, ['Selector', 'EliteSelector'])
    generator_names = get_class_names(generator_file_path, ['MoleculeGenerator'])