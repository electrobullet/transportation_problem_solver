import numpy as np
import transportation_problem as tp
from flask import Flask, redirect, render_template, request, url_for
from flask.templating import render_template_string
from flask_wtf import FlaskForm
from wtforms import BooleanField, FieldList, StringField, SubmitField

from .config import Config


class SetSizeForm(FlaskForm):
    m_field = StringField('m_field')
    n_field = StringField('n_field')
    continue_button = SubmitField('Продолжить')


class SetDataForm(FlaskForm):
    a_field = FieldList(StringField(), 'a_matrix')
    b_field = FieldList(StringField(), 'b_matrix')
    c_field = FieldList(FieldList(StringField()), 'c_matrix')
    nw_field = BooleanField('nw_field')
    continue_button = SubmitField('Продолжить')


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET', 'POST'])
def set_size():
    form = SetSizeForm(data={'m_field': '2', 'n_field': '2'})

    if form.validate_on_submit():
        return redirect(url_for('set_data', m=form.m_field.data, n=form.n_field.data))

    return render_template('set_size.html', title='Set Size', form=form)


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

    return render_template('set_data.html', title='Set Data', form=form)


@app.route('/report', methods=['GET', 'POST'])
def get_report():
    a = []
    b = []
    c = []

    for key in request.form:
        if 'a_field' in key:
            a.append(int(request.form.get(key)))
        elif 'b_field' in key:
            b.append(int(request.form.get(key)))
        elif 'c_field' in key:
            c.append(float(request.form.get(key)))

    a = np.array(a)
    b = np.array(b)
    c = np.array(c).reshape(len(a), len(b))
    use_nw_corner_method = True if request.form.get('nw_field') == 'y' else False

    html_report = tp.report.solve(tp.Data(a, b, c), use_nw_corner_method)

    with open('server/templates/base.html', encoding='utf-8') as f:
        template = f.readlines()

    html_report = ''.join(template).replace('{% block content %}{% endblock %}', html_report)

    return render_template_string(html_report, title='Report')
