from typing import Any, List

import numpy as np

from .Data import Data
from .plan import (find_cycle_path, get_start_plan_by_min_element_method,
                   get_start_plan_by_north_west_corner_method,
                   is_degenerate_plan, make_start_plan_non_degenerate,
                   recalculate_plan)


def solve(data: Data, use_nw_corner_method: bool = False) -> str:
    report_list = ['Дано:', (data.c, data.a, data.b), '']

    try:
        diff = data.get_supply_demand_difference()
        report_list.append(f'Разница между предложением и спросом: {diff}')
        report_list.append(f'Условие равновесия: {True if diff == 0 else False}')

        if diff < 0:
            data.add_dummy_supplier(-diff)
            report_list.extend([f'Добавлен фиктивный поставщик с обьемом: {-diff}', (data.c, data.a, data.b), ''])
        elif diff > 0:
            data.add_dummy_customer(diff)
            report_list.extend([f'Добавлен фиктивный потребитель с обьемом: {diff}', (data.c, data.a, data.b), ''])

        if use_nw_corner_method:
            x = get_start_plan_by_north_west_corner_method(data)
            report_list.extend(['Начальный опорный план, найденный методом северо-западного угла:',
                                (x.copy(), data.a, data.b)])
        else:
            x = get_start_plan_by_min_element_method(data)
            report_list.extend(['Начальный опорный план, полученный методом минимального элемента:',
                                (x.copy(), data.a, data.b)])

        check_res = is_degenerate_plan(x)
        report_list.extend([f'Вырожденный план: {check_res}'])
        if check_res:
            make_start_plan_non_degenerate(x)
            report_list.extend([
                '',
                'Делаем начальный опорный план невырожденным:',
                'ε - очень малое положительное число',
                (x.copy(), data.a, data.b),
            ])

        while True:
            cost = data.calculate_cost(x)
            report_list.append(f'Целевая функция: {cost}')

            p = data.calculate_potentials(x)
            report_list.append(f'Потенциалы: {p}')
            report_list.append((data.c, p, x.copy()))

            check_res = data.is_plan_optimal(x, p)
            report_list.append(f'Оптимальный план: {check_res}')
            if check_res:
                report_list.extend(['', 'Ответ:', (x.copy(), data.a, data.b), f'Целевая функция: {cost}'])
                break

            cycle_path = find_cycle_path(x, data.get_best_free_cell(x, p))
            report_list.append(f'Цикл пересчета: {cycle_path}')
            report_list.append((x.copy(), cycle_path, data.a, data.b))

            o = recalculate_plan(x, cycle_path)
            report_list.extend([f'Величина пересчета: {o}', '', 'План после пересчета:', (x.copy(), data.a, data.b)])

    finally:
        return report_list_to_html(report_list)


def report_list_to_html(report_list: List[Any]) -> str:
    html_report = [
        '<style type="text/css">',
        'body{font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;}',
        'table{text-align: center;}',
        'td{border: 1px solid black;}',
        '</style>',
    ]

    for element in report_list:
        if isinstance(element, str):
            html_report.append(f'<p>{element}</p>' if element != '' else '<hr>')

        elif isinstance(element[0], np.ndarray) and isinstance(element[1], np.ndarray) and (
            isinstance(element[2], np.ndarray)
        ):
            matrix, a, b = element
            m, n = matrix.shape

            html_report.append('<table>')

            for i in range(m + 1):
                html_report.append('<tr>')

                for j in range(n + 1):
                    if i == 0 and j == 0:
                        html_report.append('<td></td>')
                    elif i == 0 and j > 0:
                        html_report.append(f'<td style="text-align: left;">b{j} = {b[j-1]}</td>')
                    elif j == 0 and i > 0:
                        html_report.append(f'<td style="text-align: left;">a{i} = {a[i-1]}</td>')
                    else:
                        x = matrix[i-1][j-1]

                        if np.isnan(x):
                            x = 'ε'

                        html_report.append(f'<td>{x}</td>')

                html_report.append('</tr>')

            html_report.append('</table>')

        elif isinstance(element[0], np.ndarray) and isinstance(element[1], dict) and (
            isinstance(element[2], np.ndarray)
        ):
            cost, potentials, plan = element
            m, n = cost.shape

            html_report.append('<table>')

            for i in range(m + 1):
                html_report.append('<tr>')

                for j in range(n + 1):
                    if i == 0 and j == 0:
                        html_report.append('<td></td>')
                    elif i == 0 and j > 0:
                        html_report.append(f'<td style="text-align: left;">β{j} = {potentials["b"][j-1]}</td>')
                    elif j == 0 and i > 0:
                        html_report.append(f'<td style="text-align: left;">α{i} = {potentials["a"][i-1]}</td>')
                    else:
                        if plan[i-1][j-1] == 0:
                            if potentials['a'][i-1] + potentials['b'][j-1] > cost[i-1][j-1]:
                                color = 'red'
                            else:
                                color = 'lime'
                        else:
                            color = 'white'

                        html_report.append(f'<td style="background:{color};">{cost[i-1][j-1]}</td>')

                html_report.append('</tr>')

            html_report.append('</table>')

        elif isinstance(element[0], np.ndarray) and isinstance(element[1], list) and (
            isinstance(element[2], np.ndarray) and isinstance(element[3], np.ndarray)
        ):
            matrix, cycle_cells, a, b = element
            cycle_cells = cycle_cells[:-1]
            m, n = matrix.shape

            html_report.append('<table>')

            for i in range(m + 1):
                html_report.append('<tr>')

                for j in range(n + 1):
                    if i == 0 and j == 0:
                        html_report.append('<td></td>')
                    elif i == 0 and j > 0:
                        html_report.append(f'<td style="text-align: left;">b{j} = {b[j-1]}</td>')
                    elif j == 0 and i > 0:
                        html_report.append(f'<td style="text-align: left;">a{i} = {a[i-1]}</td>')
                    else:
                        color = 'white'

                        if (i-1, j-1) in cycle_cells:
                            if cycle_cells.index((i-1, j-1)) == 0:
                                color = 'yellow'
                            elif cycle_cells.index((i-1, j-1)) % 2:
                                color = 'red'
                            else:
                                color = 'lime'

                        html_report.append(f'<td style="background:{color};">{matrix[i-1][j-1]}</td>')

                html_report.append('</tr>')

            html_report.append('</table>')

    return ''.join(html_report)


def save_html_to_file(html: str, file_name: str) -> None:
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(html)
