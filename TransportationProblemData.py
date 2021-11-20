from typing import Dict, Optional, Tuple

import numpy as np

from logger import log


class TransportationProblemData:
    """
    a - обьемы поставщиков;\n
    b - обьемы потребителей;\n
    c - матрица затрат на транспортировку;\n
    r - штраф за излишки/дефицит;\n
    (все значения в ед. товара)
    """

    @log('Дано:\na: {args[1]}\nb: {args[2]}\nc:\n{args[3]}\n')
    def __init__(
        self, a: np.array, b: np.array, c: np.array, r: Optional[Dict[str, np.array]] = None,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.r = r if r is not None else {'a': np.full(self.m, 0), 'b': np.full(self.n, 0)}

    @property
    def m(self) -> int:
        """Количество поставщиков (строк матрицы c)."""
        return len(self.a)

    @property
    def n(self) -> int:
        """Количество потребителей (столбцов матрицы c)."""
        return len(self.b)

    @log('Разница между предложением и спросом: {result}')
    def get_supply_demand_difference(self) -> int:
        """Получить разницу между спросом и предложением."""
        return sum(self.a) - sum(self.b)

    @log('Добавлен фиктивный поставщик с обьемом: {args[1]}\n{args[0]}')
    def add_dummy_supplier(self, volume: int) -> None:
        """Добавить фиктивного поставщика."""
        e = np.ones(self.n) * self.r['b']
        self.c = np.row_stack((self.c, e))
        self.a = np.append(self.a, volume)

    @log('Добавлен фиктивный потребитель с обьемом: {args[1]}\n{args[0]}')
    def add_dummy_customer(self, volume: int) -> None:
        """Добавить фиктивного потребителя."""
        e = np.ones(self.m) * self.r['a']
        self.c = np.column_stack((self.c, e))
        self.b = np.append(self.b, volume)

    @log('Целевая функция: {result}')
    def calculate_cost(self, x: np.ndarray) -> float:
        """Подсчет стоимости (целевой функции)."""
        return np.sum(self.c * np.nan_to_num(x))

    @log('Потенциалы: {result}')
    def calculate_potentials(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Вычисление потенциалов."""
        res = {'a': [np.inf for _ in range(self.m)], 'b': [np.inf for _ in range(self.n)]}
        res['a'][0] = 0.0

        while np.inf in res['a'] or np.inf in res['b']:
            for i in range(self.m):
                for j in range(self.n):
                    if x[i][j] != 0:
                        if res['a'][i] != np.inf:
                            res['b'][j] = self.c[i][j] - res['a'][i]
                        elif res['b'][j] != np.inf:
                            res['a'][i] = self.c[i][j] - res['b'][j]

        return res

    @log('Оптимальный план: {result}')
    def is_plan_optimal(self, x: np.ndarray, p: Dict[str, np.ndarray]) -> bool:
        """Проверка плана на оптимальность."""
        for i, j in zip(*np.nonzero(x == 0)):
            if p['a'][i] + p['b'][j] > self.c[i][j]:
                return False

        return True

    @log('Лучшая свободная клетка для начала цикла пересчета: {result}')
    def get_best_free_cell(self, x: np.ndarray, p: Dict[str, np.ndarray]) -> Tuple[int, int]:
        free_cells = tuple(zip(*np.nonzero(x == 0)))
        return free_cells[np.argmax([p['a'][i] + p['b'][j] - self.c[i][j] for i, j in free_cells])]

    def __str__(self) -> str:
        return f'a: {self.a}\nb: {self.b}\nc:\n{self.c}'
