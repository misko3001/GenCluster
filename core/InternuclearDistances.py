from typing import Dict, Tuple, Optional, List, Set

import networkx as nx


class InternuclearDistances:
    __distances: Dict[str, Tuple[float, float]]

    def __init__(self, mol, distances: Dict[str, Tuple[float, float]]):
        new_distances = {}
        for key, value in distances.items():
            if key.count('*') == 1:
                left, right = key.split('-')
                if left == '*':
                    corrected_key = f"{right}-*"
                else:
                    corrected_key = key
                new_distances[corrected_key] = value
            else:
                new_distances[key] = value
        self.__distances = new_distances
        self.__validate(mol)

    def get(self, atom_symbol_1: str, atom_symbol_2: str) -> Optional[Tuple[float, float]]:
        value = self.__distances.get(f'{atom_symbol_1}-{atom_symbol_2}')
        if value is None:
            value = self.__distances.get(f'{atom_symbol_2}-{atom_symbol_1}')
        if value is None:
            # Wildcards
            key = f'{atom_symbol_1}-*'
            value = self.__distances.get(key)
            if value is None and atom_symbol_1 != atom_symbol_2:
                key = f'{atom_symbol_2}-*'
                value = self.__distances.get(key)
            if value is None and atom_symbol_1 != atom_symbol_2:
                key = '*-*'
                value = self.__distances.get(key)
        return value

    def __validate(self, mol) -> None:
        symbols: List[str] = mol.get_element_symbols().tolist()
        checked_bonds: Set[str] = set()
        element_graph: nx.Graph = nx.Graph()
        element_graph.add_nodes_from(set(symbols))

        symbol_length = len(symbols)
        for i in range(symbol_length):
            symbol_1 = symbols[i]
            for j in range(i, symbol_length):
                symbol_2 = symbols[j]
                if f'{symbol_1}-{symbol_2}' not in checked_bonds and f'{symbol_2}-{symbol_1}' not in checked_bonds:
                    checked_bonds.add(f'{symbol_1}-{symbol_2}')
                    distances = self.get(symbol_1, symbol_2)
                    if distances is None:
                        print(f'Warning: missing bond distances for {symbol_1}-{symbol_2}')
                    else:
                        element_graph.add_edge(symbol_1, symbol_2)

        if not nx.is_connected(element_graph):
            raise RuntimeError(f'Input molecule is missing bond distances to create a connected molecule')
