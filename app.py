import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from wtforms import FieldList, StringField, SubmitField

from config import Config
from TransportationProblemData import TransportationProblemData
from utils import solve_transportation_problem


class SetSizeForm(FlaskForm):
    m_field = StringField('m_field')
    n_field = StringField('n_field')
    continue_button = SubmitField('Продолжить')


class SetDataForm(FlaskForm):
    a_field = FieldList(StringField(), 'a_matrix')
    b_field = FieldList(StringField(), 'b_matrix')
    c_field = FieldList(FieldList(StringField()), 'c_matrix')
    continue_button = SubmitField('Продолжить')


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET', 'POST'])
def set_size():
    form = SetSizeForm(data={'m_field': '2', 'n_field': '2'})

    if form.validate_on_submit():
        return redirect(url_for('set_data', m=form.m_field.data, n=form.n_field.data))

    return render_template('set_size.html', title='Step 1', form=form)


@app.route('/data', methods=['GET', 'POST'])
def set_data():
    m = int(request.args.get('m'))
    n = int(request.args.get('n'))

    form = SetDataForm(data={
        'a_field': ['' for _ in range(m)],
        'b_field': ['' for _ in range(n)],
        'c_field': [['' for _ in range(n)] for _ in range(m)],
    })

    if form.validate_on_submit():
        return redirect(url_for('get_report'), code=307)

    return render_template('set_data.html', title='Step 2', form=form)


@app.route('/report', methods=['GET', 'POST'])
def get_report():
    a = []
    b = []
    c = []

    for key in request.form:
        if 'a_field' in key:
            a.append(request.form.get(key))
        elif 'b_field' in key:
            b.append(request.form.get(key))
        elif 'c_field' in key:
            c.append(request.form.get(key))

    a = np.array(a, np.int16)
    b = np.array(b, np.int16)
    c = np.reshape(np.array(c, np.float32), (len(a), len(b)))

    data = TransportationProblemData(a, b, c)
    report = solve_transportation_problem(data)

    return render_template('report.html', title='Report', report=report)


if __name__ == '__main__':
    app.run()
