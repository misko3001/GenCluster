from random import sample
from typing import List

from optimization.genetic.molecule.MoleculeIndividual import MoleculeIndividual as molecule
from optimization.genetic.molecule.PopulationUtils import is_better_fitness, get_best_individuals, \
    is_better_rmsd_fitness


class Selector:
    rmsd_select_limit: int = 2

    def select(self, population: list[molecule], n_ind: int) -> list[molecule]:
        raise NotImplementedError

    @staticmethod
    def check_unique_constraint(population: list[molecule], requested: int) -> None:
        if requested > len(population):
            raise ValueError(f'Cant select {requested} unique molecules from population of {len(population)} molecules')

    def check_rmsd_constraint(self, requested: int) -> None:
        if requested != self.rmsd_select_limit:
            raise ValueError(f'Rmsd selection can only be performed for {self.rmsd_select_limit} molecules')


class TournamentSelector(Selector):
    tournament_size: int

    def __init__(self, tournament_size: int = 3):
        if tournament_size < 2:
            raise ValueError('Tournament size must be greater than one')
        self.tournament_size = tournament_size

    def __str__(self) -> str:
        return f'TournamentSelector(tournament_size={self.tournament_size}, use_rmsd={self.use_rmsd})'

    def select(self, population: list[molecule], n_ind: int) -> list[molecule]:
        self.check_unique_constraint(population, n_ind)
        selected: [molecule] = []
        winners: {molecule} = set()
        for i in range(n_ind):
            candidates = [mol for mol in population if mol not in winners]
            winner = self.tournament_selection(candidates, self.tournament_size)
            selected.append(winner)
            winners.add(winner)
        return selected

    @staticmethod
    def tournament_selection(population: list[molecule], tournament_size: int) -> molecule:
        if len(population) < tournament_size:
            tournament_size = len(population)
        candidates: [molecule] = sample(population, tournament_size)
        winner: molecule = candidates[0]
        for candidate in candidates[1:]:
            if is_better_fitness(candidate, winner):
                winner = candidate
        return winner


class RmsdTournamentSelector(TournamentSelector):
    tournament_size: int
    rmsd_methods: List[str]

    def __init__(self, rmsd_methods: List[str], tournament_size: int = 3):
        super().__init__(tournament_size)
        self.rmsd_methods = rmsd_methods

    def __str__(self) -> str:
        return f'RmsdTournamentSelector(tournament_size={self.tournament_size})'

    def select(self, population: list[molecule], n_ind: int = 2) -> list[molecule]:
        self.check_unique_constraint(population, n_ind)
        self.check_rmsd_constraint(n_ind)
        selected: [molecule] = []
        winners: {molecule} = set()
        for i in range(n_ind):
            candidates = [mol for mol in population if mol not in winners]
            if len(winners) == 0:
                winner: molecule = self.tournament_selection(candidates, self.tournament_size)
            else:
                winner = self.rmsd_tournament_selection(candidates, self.tournament_size, selected[0], self.rmsd_methods)
            selected.append(winner)
            winners.add(winner)
        return selected

    @staticmethod
    def rmsd_tournament_selection(population: list[molecule], tournament_size: int, reference: molecule,
                                  methods: List[str]) -> molecule:
        if len(population) < tournament_size:
            tournament_size = len(population)
        candidates: [molecule] = sample(population, tournament_size)
        winner: molecule = candidates[0]
        winner_rmsd = candidates[0].get_rmsd(reference, methods)
        for candidate in candidates[1:]:
            candidate_rmsd = candidate.get_rmsd(reference, methods)
            if is_better_rmsd_fitness(candidate, winner, candidate_rmsd, winner_rmsd):
                winner = candidate
                winner_rmsd = candidate_rmsd
        return winner


class EliteSelector(Selector):
    elite_quantity: int

    def __init__(self, elite_quantity: int = 1):
        if elite_quantity < 1:
            raise ValueError('Number of elites must be greater than zero.')
        self.elite_quantity = elite_quantity

    def __str__(self) -> str:
        return f'EliteSelector(elite_quantity={self.elite_quantity})'

    def select(self, population: list[molecule], n_ind: int = 0) -> list[molecule]:
        self.check_unique_constraint(population, self.elite_quantity)
        return self.elite_selection(population, self.elite_quantity)

    @staticmethod
    def elite_selection(population: list[molecule], n_ind: int) -> list[molecule]:
        return get_best_individuals(population, n_ind)
