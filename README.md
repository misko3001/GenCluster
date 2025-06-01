# Usage

GenCluster was built and tested with Python 3.11
(should also work with version 3.12)

The recommended way of installation is by creating a virtual environment using venv. The instructions are outlined in the following steps:

1. Download and extract the project from the archive
2. Set up the virtual environment using venv
    1. Navigate to the folder where the project is located (the file `main.py`)
    2. Create the environment: `python3 -m venv .venv`
    3. Activate the environment: `source .venv/bin/activate`
    4. Install the required libraries: `pip install -r requirements.txt`
3. Run the program: `python3 main.py genetic -config <path>`

For future runs, you only need to activate the previously created virtual environment using the same command and run main.py.
The application includes example files located in the folder `data/examples`.
To run one of the examples, use the following command:

>python3 main.py genetic -config data/examples/Au_6/au_6.yaml

# Manual

Describes the basic components of the algorithm.

## Configuration files 

The application uses input YAML configuration files as the only required input for the algorithm. During execution,
the configuration file is specified using the `-config` flag along with the file path.
These files contain all the necessary settings for the input problem, parallelization, model selection,
and genetic operators. A created configuration file can be easily saved for later use or sharing.

>Documentation for each setting is available in the file `data/examples/example_config.yaml`.

The configuration file is divided into 9 main groups. Each group configures a different part of the algorithm, and their descriptions are listed below:

* `problem` – sets general parameters for the algorithm and input
* `population` – configures the genetic algorithm population
* `optimizer` – configures the model and optimization parameters
* `evaluator` – configures the model used to compute potential energy
* `generator` – defines the method for generating clusters
* `selector` – selects the selection operator and its parameters
* `crossover` – selects the crossover operator and its parameters
* `mutators` – selects a group of mutation operators and their parameters
* `terminators` – defines the termination conditions of the algorithm

The documentation file provides more detailed descriptions of each setting. Examples of real configuration files can be found in `data/examples`.

>The application includes validation of configuration files to prevent issues during their creation.
If the file is invalid, the program stops at the beginning of the algorithm and informs the user where the error occurred.

## Integration with MLatom (ORCA and xtb)

The MLatom library is primarily used for energy calculations and performing local optimizations.
One of the features of this library is providing interfaces to chemical computational software, which are utilized
in the application. The supported external programs are XTB, ORCA, and the PySCF library
(written directly in Python). The calls to these programs are implemented in the `GeometryOptimizer` class.

To use the two external programs (XTB and ORCA), they must be installed on the device running the GenCluster program.
After installation, you also need to define a system environment variable that contains the path to the program's directory.
These variables can be set using the following commands:

* ORCA: `export orcabin="/<path-to-ORCA>/orca"`
* XTB: `export xtb="/<path-to-XTB>/bin/xtb"`

## Adding new configuration settings

New configurations can be added to the configuration file in the following way (example for population):

1. In the file `core/ConfigParser.py`, there are classes that correspond to the setting names in the YAML configuration file.
2. To add a new configuration setting for `population`, you need to find the class `PopulationConfig` and add a new variable to it that corresponds to the name of the new configuration.
3. For comprehensive input value validation, you can create a function with the annotation `@field_validator('<variable-name>', mode='before/after')`.
4. Extend the existing documentation to include the newly added variable (in `data/examples/example_config.yaml`).
5. The new variable will then be available in the file (and class) `core/EvolutionContext.py`, and it can be accessed as `config.population.<variable-name>`.

>The `EvolutionContext` class initializes the base class `EvolutionEngine`, which manages the genetic algorithm. In this class, all parts of the genetic algorithm can be configured, such as the genetic operators used.

## Adding new genetic operators, generators or terminating conditions

Creating new genetic operators, generators, and termination conditions is a relatively simple process.
For example, to add a new selection operator, you need to create a new class by extending the base class `Selector`
located in the file `optimization/genetic/operations/Selector.py`. This base class contains a single function `select`,
which must be implemented by the user.
>The application automatically detects the new class in the file, and it can be referenced by its class name in the input configuration file.

If your new addition requires additional variables, you need to add them to the configuration file following the same procedure
described in the previous section (Adding new configuration settings).
Existing operators can serve as a template for implementation.

Once created, the new class must be initialized with the required variables in the `EvolutionContext` class.

Other base classes (`Crossover`, `Mutator`, and `Terminator`) can be found in the `optimization/genetic/operations`
directory. The base class `MoleculeGenerator` is located in `optimization/genetic/molecule/MoleculeGenerator.py`.
The process for creating new implementations of these classes is similar to the process for creating a new selection operator.

## Adding new RMSD methods

RMSD and alignment methods are located by default in the file `optimization/genetic/molecule/MoleculeUtils.py`.

New methods can be added using the following steps:
1. Create and implement a new function in `MoleculeUtils.py`.
2. Add a new identifier for the created function to the variable `AlignmentMethods` or `RMSDMethods`
in the same file (these global variables are located at the top of the file).
3. Add a case for the new identifier in the `MoleculeIndividual` class within the `__align` or `__calculate_rmsd` functions.
4. The created identifier can then be used in the input configuration file under `problem.alignment_methods` or `problem.rmsd_methods`.