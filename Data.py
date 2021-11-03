from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger import logger


class Data:
    """
    a - обьемы поставщиков;\n
    b - обьемы потребителей;\n
    c - матрица затрат на транспортировку единицы товара;\n
    r - штраф за излишки/дефицит единицы товара;\n
    """

    def __init__(
        self,
        a: List[int],
        b: List[int],
        c: np.ndarray,
        r: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.r = {'a': [0 for _ in range(self.m)], 'b': [0 for _ in range(self.n)]}

        logger.info(f'Дано:\na = {a}\nb = {b}\nc = \n{c}\nr = {r}')

    @property
    def m(self) -> int:
        """Количество поставщиков (строк матрицы c)."""
        return len(self.a)

    @property
    def n(self) -> int:
        """Количество потребителей (столбцов матрицы c)."""
        return len(self.b)

    def get_supply_demand_difference(self) -> int:
        """Получить разницу между спросом и предложением."""
        return sum(self.a) - sum(self.b)

    def add_dummy_supplier(self, diff: int) -> None:
        """Добавить фиктивного поставщика."""
        e = np.ones((1, self.c.shape[1])) * self.r['b']
        self.c = np.row_stack((self.c, e))
        self.a.append(-diff)

    def add_dummy_customer(self, diff: int) -> None:
        """Добавить фиктивного потребителя."""
        e = np.ones((self.c.shape[0], 1)) * self.r['a']
        self.c = np.column_stack((self.c, e))
        self.b.append(diff)

    def do_min_element_method(self) -> np.ndarray:
        """Получить начальный опорный план методом минимального элемента."""
        def get_min_element_position(matrix: np.ndarray) -> Tuple[int, int]:
            """Получить позицию минимального элемента матрицы."""
            flat_index = np.argmin(matrix)
            i = flat_index // self.m
            j = flat_index - i * self.n
            return (i, j)

        a = self.a.copy()
        b = self.b.copy()
        c = self.c.copy()
        res = np.zeros((self.m, self.n))

        has_dummy_row = False
        has_dummy_column = False

        if sum(c[-1]) == 0:
            c[-1] = np.inf
            has_dummy_row = True
        elif sum(c[:, -1]) == 0:
            c[:, -1] = np.inf
            has_dummy_column = True

        while sum(a) + sum(b) != 0:
            i, j = get_min_element_position(c)
            c[i][j] = np.inf

            x = min(a[i], b[j])
            a[i] -= x
            b[j] -= x

            res[i][j] = x

            if a[i] == 0:
                c[i] = np.inf

            if b[j] == 0:
                c[:, j] = np.inf

            if np.min(c) == np.inf:
                if has_dummy_row:
                    c[-1] = 0
                elif has_dummy_column:
                    c[:, -1] = 0

        return res

    def do_north_west_corner_method(self) -> np.ndarray:
        """Получить начальный опорный план методом северо-западного угла."""
        a = self.a.copy()
        b = self.b.copy()
        res = np.zeros((self.m, self.n))
        i = 0
        j = 0

        while i < self.m and j < self.n:
            x = min(a[i], b[j])
            a[i] -= x
            b[j] -= x

            res[i][j] = x

            if a[i] == 0:
                i += 1

            if b[j] == 0:
                j += 1

        return res

    def get_start_plan(self, mode: int = 0) -> np.ndarray:
        if mode == 0:
            return self.do_min_element_method()
        elif mode == 1:
            return self.do_north_west_corner_method()
        else:
            raise ValueError('The mode must be 0 or 1.')

    def is_degenerate_plan(self) -> bool:
        """Проверка плана на вырожденность."""
        return True if np.count_nonzero(self.x) != self.m + self.n - 1 else False

    def calculate_potentials(self) -> Dict[str, List[Any]]:
        """Вычисление потенциалов."""
        potentials = {'a': [None for _ in range(self.m)], 'b': [None for _ in range(self.n)]}
        potentials['a'][0] = 0  # type: ignore

        while None in potentials['a'] or None in potentials['b']:
            for i in range(self.m):
                for j in range(self.n):
                    if self.x[i][j] != 0:
                        if potentials['a'][i] is not None:
                            potentials['b'][j] = self.c[i][j] - potentials['a'][i]
                        elif potentials['b'][j] is not None:
                            potentials['a'][i] = self.c[i][j] - potentials['b'][j]

        return potentials

    def is_plan_optimal(self) -> bool:
        """Проверка плана на оптимальность."""
        for i, j in zip(*np.nonzero(self.x == 0)):
            if self.p['a'][i] + self.p['b'][j] > self.c[i][j]:
                return False

        return True

    def do_cycle(self) -> List[Tuple[int, int]]:
        """Построение цикла пересчета."""
        basic_cells = tuple(zip(*np.nonzero(self.x)))
        free_cells = tuple(zip(*np.nonzero(self.x == 0)))

        cycle_cells = []

        for i, j in free_cells:
            if self.p['a'][i] + self.p['b'][j] > self.c[i][j]:
                cycle_cells.append((i, j))

        while len(cycle_cells) != len(basic_cells) + 1:
            for i, j in basic_cells:
                if (i == cycle_cells[-1][0] or j == cycle_cells[-1][1]) and (i, j) not in cycle_cells:
                    cycle_cells.append((i, j))

        return cycle_cells

    def recalculate_plan(self, cycle_cells: List[Tuple[int, int]]):
        o = min([self.x[i][j] for i, j in cycle_cells[1:]])

        for n, (i, j) in enumerate(cycle_cells):
            if n % 2 == 0:
                self.x[i][j] += o
            else:
                self.x[i][j] -= o

    def calculate_cost(self) -> float:
        """Подсчет стоимости (целевой функции)."""
        return np.sum(self.c * self.x)

    def solve(self) -> np.ndarray:
        diff = self.get_supply_demand_difference()
        logger.info(f'Разница между предложением и спросом = {diff}')

        if diff == 0:
            logger.info('Условие равновесия выполняется')
        else:
            logger.info('Условие равновесия не выполняется')

        if diff < 0:
            self.add_dummy_supplier(diff)
            logger.info(f'Добавлен фиктивный пункт A{self.m} с обьемом = {-diff}')
        elif diff > 0:
            self.add_dummy_customer(diff)
            logger.info(f'Добавлен фиктивный пункт B{self.n} с обьемом = {diff}')

        self.x = self.get_start_plan()
        logger.info(f'Начальный опорный план, найденный методом северо-западного угла:\n{self.x}')
        logger.info(f'Целевая функция = {self.calculate_cost()}')

        if self.is_degenerate_plan():
            logger.info('План - вырожденный')
        else:
            logger.info('План - невырожденный')

        self.p = self.calculate_potentials()
        logger.info(f'Найденные потенциалы: {self.p}')

        if self.is_plan_optimal():
            logger.info('План оптимален')
            return self.x
        else:
            logger.info('План неоптимален')

        cycle_cells = self.do_cycle()
        self.recalculate_plan(cycle_cells)

        logger.info(f'{self.x}')

        self.p = self.calculate_potentials()
        logger.info(f'Найденные потенциалы: {self.p}')

        if self.is_plan_optimal():
            logger.info('План оптимален')
        else:
            logger.info('План неоптимален')

        logger.info(f'Целевая функция = {self.calculate_cost()}')

        return self.x


if __name__ == '__main__':
    data = Data(
        a=[4, 6, 8],
        b=[3, 6, 5, 7],
        c=np.array([
            [2, 4, 1, 3],
            [4, 8, 2, 4],
            [2, 2, 6, 5],
        ]),
    )
    # L = 41

    data.solve()
