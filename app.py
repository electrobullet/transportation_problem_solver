import remi.gui as gui
from remi import App, start


class MyApp(App):
    def __init__(self, *args) -> None:
        super(MyApp, self).__init__(*args)

    def main(self) -> gui.Widget:
        self.num_of_rows = 3
        self.num_of_columns = 4

        grid_box = gui.GridBox()
        self.get_start_layout(grid_box)

        return grid_box

    def get_matrix_layout(self, grid_box: gui.GridBox) -> None:
        widgets = {
            f'c_{i}_{j}': gui.Input()
            for i in range(self.num_of_rows)
            for j in range(self.num_of_columns)
        }
        widgets['button'] = gui.Button('Compute')

        grid_box.empty()
        grid_box.append(widgets)

        layout = [
            [f'c_{i}_{j}' for j in range(self.num_of_columns)]
            for i in range(self.num_of_rows)
        ]
        layout.append(['button'] * self.num_of_columns)

        grid_box.define_grid(layout)

    def get_start_layout(self, grid_box: gui.GridBox) -> None:
        def on_button_click() -> None:
            self.num_of_rows = int(widgets['num_of_rows'].get_value())
            self.num_of_columns = int(widgets['num_of_columns'].get_value())

            self.get_matrix_layout(grid_box)

        widgets = {
            'row_label': gui.Label('Number of rows (suppliers)'),
            'num_of_rows': gui.Input(default_value=self.num_of_rows),
            'column_label': gui.Label('Number of columns (clients)'),
            'num_of_columns': gui.Input(default_value=self.num_of_columns),
            'button': gui.Button('Continue'),
        }

        widgets['button'].onclick.do(lambda _: on_button_click())

        grid_box.empty()
        grid_box.append(widgets)

        layout = [
            ['row_label', 'num_of_rows'],
            ['column_label', 'num_of_columns'],
            ['button', 'button'],
        ]

        grid_box.define_grid(layout)


if __name__ == '__main__':
    start(MyApp, debug=True, address='0.0.0.0', port=8001, start_browser=False)
