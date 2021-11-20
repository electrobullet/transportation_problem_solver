import os

import numpy as np

import transportation_problem as tp

if __name__ == '__main__':
    data = tp.Data(
        np.array([12, 30, 13]),
        np.array([23, 40, 12, 32]),
        np.array([
            [64, 32, 45, 12],
            [32, 78, 23, 90],
            [88, 67, 10, 32],
        ]),
    )

    tp.report.save_html_to_file(tp.solve(data), 'report.html')
    os.startfile('report.html')
