from typing import List, Tuple

import numpy as np

from logger import log, logger
from TransportationProblemData import TransportationProblemData


@log('\nНачальный опорный план, полученный методом минимального элемента:\n{result}')
def get_start_plan_by_min_element_method(data: TransportationProblemData) -> np.ndarray:
    """Получить начальный опорный план методом минимального элемента."""
    def get_min_element_position(matrix: np.ndarray) -> Tuple[int, int]:
        """Получить позицию минимального элемента матрицы."""
        flat_index = np.argmin(matrix)
        i = flat_index // data.m
        j = flat_index - i * data.n
        return (i, j)

    res = np.zeros((data.m, data.n))

    a = data.a.copy()
    b = data.b.copy()
    c = data.c.copy()

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


@log('\nНачальный опорный план, найденный методом северо-западного угла:\n{result}')
def get_start_plan_by_north_west_corner_method(data: TransportationProblemData) -> np.ndarray:
    """Получить начальный опорный план методом северо-западного угла."""
    res = np.zeros((data.m, data.n))

    a = data.a.copy()
    b = data.b.copy()
    i = 0
    j = 0

    while i < data.m and j < data.n:
        x = min(a[i], b[j])
        a[i] -= x
        b[j] -= x

        res[i][j] = x

        if a[i] == 0:
            i += 1

        if b[j] == 0:
            j += 1

    return res


@log('Вырожденный план: {result}')
def is_degenerate_plan(x: np.ndarray) -> bool:
    """Проверка плана на вырожденность."""
    m, n = x.shape
    return True if np.count_nonzero(x) != m + n - 1 else False


@log('Цикл пересчета: {result}')
def find_cycle_path(x: np.ndarray, start_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    def get_posible_moves(bool_table: np.ndarray, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        posible_moves = np.full(bool_table.shape, False)

        current_pos = path[-1]
        prev_pos = path[-2] if len(path) > 1 else (np.nan, np.nan)

        if current_pos[0] != prev_pos[0]:
            posible_moves[current_pos[0]] = True

        if current_pos[1] != prev_pos[1]:
            posible_moves[:, current_pos[1]] = True

        return list(zip(*np.nonzero(posible_moves * bool_table)))

    res = [start_pos]
    bool_table = x != 0

    while True:
        current_pos = res[-1]

        bool_table[current_pos[0]][current_pos[1]] = False

        if len(res) > 3:
            bool_table[start_pos[0]][start_pos[1]] = True

        posible_moves = get_posible_moves(bool_table, res)

        if start_pos in posible_moves:
            res.append(start_pos)
            return res

        if not posible_moves:
            for i, j in res[1:-1]:
                bool_table[i][j] = True

            res = [start_pos]
            continue

        res.append(posible_moves[0])


@log('Величина пересчета: {result}\n\nПлан после пересчета:\n{args[0]}')
def recalculate_plan(x: np.ndarray, cycle_path: List[Tuple[int, int]]) -> int:
    """Пересчитать план. Возвращает величину пересчета."""
    o = min([x[i][j] for i, j in cycle_path[1:-1:2]])

    for k, (i, j) in enumerate(cycle_path[:-1]):
        if np.isnan(o):
            if x[i][j] == 0:
                x[i][j] = np.nan
            elif np.isnan(x[i][j]):
                x[i][j] = 0

            continue

        if k % 2 == 0:
            x[i][j] += o
        else:
            x[i][j] -= o

    return o


@log('\nДелаем начальный опорный план невырожденным:\n{args[0]}')
def make_start_plan_non_degenerate(x: np.ndarray) -> None:
    for row in x:
        if np.count_nonzero(row) == 1:
            row[np.nonzero(row == 0)[0][0]] = np.nan
            break


def solve_transportation_problem(data: TransportationProblemData, use_nw_corner_method: bool = False):
    logger.info(f'Дано:\n{data}\n')

    diff = data.get_supply_demand_difference()

    if diff < 0:
        data.add_dummy_supplier(-diff)
    elif diff > 0:
        data.add_dummy_customer(diff)

    if use_nw_corner_method:
        x = get_start_plan_by_north_west_corner_method(data)
    else:
        x = get_start_plan_by_min_element_method(data)

    if is_degenerate_plan(x):
        make_start_plan_non_degenerate(x)

    while True:
        data.calculate_cost(x)

        p = data.calculate_potentials(x)

        if data.is_plan_optimal(x, p):
            return x

        cycle_path = find_cycle_path(x, data.get_best_free_cell(x, p))
        recalculate_plan(x, cycle_path)

        if is_degenerate_plan(x):
            return


if __name__ == '__main__':
    data = TransportationProblemData(
        a=[4, 6, 8],
        b=[3, 6, 5, 7],
        c=[
            [2, 4, 1, 3],
            [4, 8, 2, 4],
            [2, 2, 6, 5],
        ],
        r={'a': [0, 0, 0], 'b': [1, 2, 3, 4]}
    )

    solve_transportation_problem(data)
