from typing import Any, List

import numpy as np


def get_html_report(report_list: List[Any]) -> str:
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
                        html_report.append(f'<td>{matrix[i-1][j-1]}</td>')

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


def save_html_string_to_file(html: str, file_name: str) -> None:
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(html)


def insert_html_into_template(html: str, template_name: str) -> str:
    with open(template_name, encoding='utf-8') as f:
        template = f.readlines()

    return ''.join(template).replace('{% block content %}{% endblock %}', html)
