from typing import List, Tuple

import numpy as np

from .Data import Data


def get_start_plan_by_min_element_method(data: Data) -> np.ndarray:
    """Получить начальный опорный план методом минимального элемента."""
    def get_min_element_position(matrix: np.ndarray) -> Tuple[int, int]:
        """Получить позицию минимального элемента матрицы."""
        flat_index = np.argmin(matrix)
        i = flat_index // data.n
        j = flat_index - i * data.n
        return (i, j)

    res = np.zeros((data.m, data.n))

    a = data.a.copy()
    b = data.b.copy()
    c = data.c.copy()

    if data.has_dummy_row:
        c[-1] = np.inf
    elif data.has_dummy_column:
        c[:, -1] = np.inf

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
            if data.has_dummy_row:
                c[-1] = 0
            elif data.has_dummy_column:
                c[:, -1] = 0

    return res


def get_start_plan_by_north_west_corner_method(data: Data) -> np.ndarray:
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


def is_degenerate_plan(x: np.ndarray) -> bool:
    """Проверка плана на вырожденность."""
    m, n = x.shape
    return True if np.count_nonzero(x) != m + n - 1 else False


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


def recalculate_plan(x: np.ndarray, cycle_path: List[Tuple[int, int]]) -> int:
    """Пересчитать план. Возвращает величину пересчета."""
    o = np.min([x[i][j] for i, j in cycle_path[1:-1:2]])
    minus_cells_equal_to_o = [(i, j) for i, j in cycle_path[1:-1:2] if np.isnan(x[i][j]) or x[i][j] == o]

    if np.isnan(o):
        i, j = cycle_path[0]
        x[i][j] = np.nan
        i, j = minus_cells_equal_to_o[0]
        x[i][j] = 0

        return o

    for k, (i, j) in enumerate(cycle_path[:-1]):
        if (i, j) in minus_cells_equal_to_o:
            if minus_cells_equal_to_o.index((i, j)) == 0:
                x[i][j] = 0
            else:
                x[i][j] = np.nan

            continue

        if np.isnan(x[i][j]):
            x[i][j] = 0

        if k % 2 == 0:
            x[i][j] += o
        else:
            x[i][j] -= o

    return o


def make_start_plan_non_degenerate(x: np.ndarray) -> None:
    m, n = x.shape

    while np.count_nonzero(x) != m + n - 1:
        for i in range(m):
            if np.count_nonzero(x[i]) == 1:
                j = np.nonzero(x[i])[0][0]

                if np.count_nonzero(x[:, j]) == 1:
                    if np.nonzero(x[:, j])[0][0] == i:
                        if i < m - 1:
                            x[i + 1][j] = np.nan
                        else:
                            x[i - 1][j] = np.nan

                        break
