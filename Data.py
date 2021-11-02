from typing import Dict, List, Optional

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

    def get_start_plan(self) -> np.ndarray:
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

    def is_degenerate_plan(self) -> bool:
        """Проверка плана на вырожденность"""
        return True if np.count_nonzero(self.x) != self.m + self.n - 1 else False

    def calculate_cost(self) -> float:
        """Подсчет стоимости (целевой функции)."""
        return np.sum(self.c * self.x)

    def solve(self) -> np.ndarray:
        diff = self.get_supply_demand_difference()
        logger.info(f'Разница между предложением и спросом = {diff}')

        if diff < 0:
            self.add_dummy_supplier(diff)
            logger.info(f'Добавлен фиктивный пункт A{self.m} с обьемом = {-diff}')
        elif diff > 0:
            self.add_dummy_customer(diff)
            logger.info(f'Добавлен фиктивный пункт B{self.n} с обьемом = {diff}')

        self.x = self.get_start_plan()
        logger.info(f'Начальный опорный план, найденный методом северо-западного угла:\n{self.x}')

        if self.is_degenerate_plan():
            logger.info('План - вырожденный')
        else:
            logger.info('План - невырожденный')

        logger.info(f'Целевая функция = {self.calculate_cost()}')

        return self.x


if __name__ == '__main__':
    data = Data(
        a=[4, 5, 4, 5],
        b=[3, 8, 4, 3],
        c=np.array([
            [3, 4, 1, 2],
            [2, 2, 4, 3],
            [1, 1, 2, 1],
            [1, 1, 1, 1],
        ]),
    )

    data.solve()
