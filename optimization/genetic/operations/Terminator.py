import itertools
from collections import deque
from typing import Optional, List
from datetime import datetime, timedelta

from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule


class Terminator:

    def check_condition(self, **kwargs) -> bool:
        raise NotImplementedError

    def get_reason(self) -> str:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__

    def get_terminator_values(self) -> Optional[List[any]]:
        return None

    def update_terminator_values(self, debug_values: List[any]):
        raise NotImplementedError


class MaxIterationsTerminator(Terminator):
    max_iterations: int

    def __init__(self, max_iterations: int):
        if max_iterations < 1:
            raise ValueError("Max iterations must be greater than or equal to 1")
        self.max_iterations = max_iterations

    def __str__(self) -> str:
        return f'MaxIterationsTerminator(max_iterations={self.max_iterations})'

    def check_condition(self, **kwargs) -> bool:
        if 'iteration' not in kwargs:
            raise ValueError("Missing iteration argument for MaxIterationsTerminator")
        iteration = kwargs['iteration']
        return iteration >= self.max_iterations

    def get_reason(self) -> str:
        return f'Reached maximum number of iterations: {self.max_iterations}'


class StagnationTerminator(Terminator):
    best: Optional[float] = None
    best_iterations: int = 0
    max_stagnating_iterations: int = 0

    def __init__(self, max_stagnating_iterations: int):
        if max_stagnating_iterations < 1:
            raise ValueError("Max stagnating iterations with the same best molecule must be greater than or equal to 1")
        self.max_stagnating_iterations = max_stagnating_iterations

    def __str__(self) -> str:
        return f'StagnationTerminator(max_stagnating_iterations={self.max_stagnating_iterations})'

    def check_condition(self, **kwargs) -> bool:
        if 'new_best' not in kwargs:
            raise ValueError("Missing new_best argument for StagnationTerminator")
        new_best: Optional[molecule] = kwargs['new_best']
        self.update_best(new_best)
        return self.best_iterations >= self.max_stagnating_iterations

    def update_best(self, new_best: Optional[molecule]) -> None:
        if new_best is None:
            self.best_iterations += 1
        elif self.best is None or new_best.get_fitness() < self.best:
            self.best = new_best.get_fitness()
            self.best_iterations = 0
        else:
            self.best_iterations += 1

    def get_reason(self) -> str:
        return f'Reached maximum number of stagnating iterations (with energy: {self.best})'

    def get_terminator_values(self) -> Optional[List[any]]:
        return [self.best, self.best_iterations]

    def update_terminator_values(self, debug_values: List[any]):
        self.best = debug_values[0]
        self.best_iterations = debug_values[1]


class DurationTerminator(Terminator):
    TIME_UNITS = ['seconds', 'minutes', 'hours']
    start: datetime
    end: Optional[datetime] = None
    delta: timedelta
    unit_type: str
    unit_value: float

    def __init__(self, unit_type: str, unit_value: float, start: datetime):
        if unit_type is None:
            raise ValueError("Unit type cannot be None")
        elif unit_type.casefold() not in self.TIME_UNITS:
            raise ValueError(f"Unit type '{unit_type}' is not supported")
        if unit_value is None:
            raise ValueError("Unit value cannot be None")
        elif unit_value <= 0:
            raise ValueError("Unit value must be greater than 0")
        if start is None:
            raise ValueError("Start date cannot be None")
        self.unit_type = unit_type.casefold()
        self.unit_value = unit_value
        self.start = start
        self.delta = self.__get_timedelta()

    def __str__(self) -> str:
        return f'DurationTerminator(unit_type={self.unit_type}, unit_value={self.unit_value}, start={self.start})'

    def check_condition(self, **kwargs) -> bool:
        current = datetime.now()
        if current - self.start >= self.delta:
            self.end = current
            return True
        return False

    def get_reason(self) -> str:
        return f'Reached time limit {self.unit_value} {self.unit_type} (start: {self.start}, end: {self.end})'

    def get_terminator_values(self) -> Optional[List[any]]:
        return [self.unit_type, self.unit_value, (datetime.now() - self.start).total_seconds()]

    def update_terminator_values(self, debug_values: List[any]):
        self.unit_type = debug_values[0]
        self.unit_value = debug_values[1]
        elapsed_time = debug_values[2]
        self.start = datetime.now() - timedelta(seconds=elapsed_time)

    def __get_timedelta(self) -> timedelta:
        match self.unit_type:
            case 'seconds':
                return timedelta(seconds=self.unit_value)
            case 'minutes':
                return timedelta(minutes=self.unit_value)
            case 'hours':
                return timedelta(hours=self.unit_value)
            case _:
                raise ValueError(f"Unsupported unit type '{self.unit_type}'")


class ConvergenceTerminator(Terminator):
    last_m_generations: int
    last_n_generations: int
    delta: float
    last_best: deque[float]

    def __init__(self, last_m_generations: int, last_n_generations: int, delta: float):
        if last_m_generations is None:
            raise ValueError("last_m_generations cannot be None")
        elif last_m_generations < 2:
            raise ValueError("last_m_generations must be greater than or equal to 2")
        if last_n_generations is None:
            raise ValueError("last_n_generations cannot be None")
        elif last_n_generations < 1:
            raise ValueError("last_n_generations must be greater than or equal to 1")
        elif last_m_generations <= last_n_generations:
            raise ValueError("last_m_generations must be greater than last_n_generations")
        if delta is None:
            raise ValueError("delta cannot be None")
        elif delta <= 0:
            raise ValueError("delta must be greater than 0")
        self.last_m_generations = last_m_generations
        self.last_n_generations = last_n_generations
        self.delta = delta
        self.last_best = deque(maxlen=last_m_generations)

    def __str__(self) -> str:
        return (f'ConvergenceTerminator(last_m_generations={self.last_m_generations}, '
                f'last_n_generations={self.last_n_generations}, delta={self.delta})')

    def check_condition(self, **kwargs) -> bool:
        if 'new_best' not in kwargs:
            raise ValueError("Missing new_best argument for ConvergenceTerminator")

        new_best: Optional[molecule] = kwargs['new_best']
        self.__update_last_best(new_best)

        if self.last_m_generations != len(self.last_best):
            return False

        return self.__get_average() < self.delta

    def __get_average(self):
        return abs(self.__get_m_average() - self.__get_n_average())

    def __get_m_average(self):
        return sum(self.last_best) / self.last_m_generations

    def __get_n_average(self):
        return sum(itertools.islice(self.last_best, self.last_n_generations)) / self.last_n_generations

    def __update_last_best(self, new_best: Optional[molecule]) -> None:
        if new_best is not None:
            if len(self.last_best) == self.last_m_generations:
                self.last_best.pop()
            self.last_best.appendleft(new_best.get_fitness())

    def get_reason(self) -> str:
        return (f'Reached convergence delta {self.delta} (current delta: {self.__get_average()}, '
                f'm generations average: {self.__get_m_average()}, n generations average: {self.__get_n_average()})')

    def get_terminator_values(self) -> Optional[List[any]]:
        return [self.last_m_generations, self.last_n_generations, self.delta, list(self.last_best)]

    def update_terminator_values(self, debug_values: List[any]):
        self.last_m_generations = debug_values[0]
        self.last_n_generations = debug_values[1]
        self.delta = debug_values[2]
        last_best_list = debug_values[3]
        self.last_best = deque(last_best_list, maxlen=self.last_m_generations)
