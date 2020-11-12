# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import json
import math
import operator
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import librosa
import numpy as np
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# number of items in a table per page
DATA_PAGE_SIZE = 10

# operators for filtering items
filter_operators = [
    ['ge ', '>='],
    ['le ', '<='],
    ['lt ', '<'],
    ['gt ', '>'],
    ['ne ', '!='],
    ['eq ', '='],
    ['contains '],
]

# parse table filter queries
def split_filter_part(filter_part):
    for operator_type in filter_operators:
        for op in operator_type:
            if op in filter_part:
                name_part, value_part = filter_part.split(op, 1)
                name = name_part[name_part.find('{') + 1 : name_part.rfind('}')]
                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1:-1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                return name, operator_type[0].strip(), value
    return [None] * 3


# standard command-line arguments parser
def parse_args():
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    return args


# load data from JSON manifest file
def load_data(data_filename):
    data = []
    num_hours = 0.0
    vocabulary = defaultdict(lambda: 0)
    alphabet = set()
    with open(data_filename, 'r', encoding='utf8') as f:
        for line in f:
            item = json.loads(line)
            num_words = len(item['text'].split())
            num_chars = len(item['text'])
            data.append(
                {
                    'audio_filepath': item['audio_filepath'],
                    'duration': round(item['duration'], 2),
                    'num_words': num_words,
                    'num_chars': num_chars,
                    'word_rate': round(num_words / item['duration'], 2),
                    'char_rate': round(num_chars / item['duration'], 2),
                    'text': item['text'],
                }
            )
            for k in item:
                if k not in data[-1]:
                    data[-1][k] = item[k]
            num_hours += item['duration']
            for word in item['text'].split():
                vocabulary[word] += 1
            for char in item['text']:
                alphabet.add(char)
    num_hours /= 60.0 * 60.0
    vocabulary_data = [{'word': word, 'count': vocabulary[word]} for word in vocabulary]
    return data, num_hours, vocabulary_data, alphabet


# plot histogram of specified field in data list
def plot_histogram(data, key, label):
    fig = px.histogram(
        data_frame=[item[key] for item in data],
        nbins=50,
        log_y=True,
        labels={'value': label},
        opacity=0.5,
        color_discrete_sequence=['green'],
        height=200,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
    return fig


args = parse_args()
print('Loading data...')
data, num_hours, vocabulary, alphabet = load_data(args.manifest)
print('Starting server...')
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

figure_duration = plot_histogram(data, 'duration', 'Duration (sec)')
figure_num_words = plot_histogram(data, 'num_words', '#words')
figure_num_chars = plot_histogram(data, 'num_chars', '#chars')
figure_word_rate = plot_histogram(data, 'word_rate', '#words/sec')
figure_char_rate = plot_histogram(data, 'char_rate', '#chars/sec')

stats_layout = [
    dbc.Row(dbc.Col(html.H5(children='Global Statistics'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        [
            dbc.Col(html.Div('Number of hours', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Number of utterances', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Vocabulary size', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Alphabet size', className='text-secondary'), width=3),
        ],
        className='bg-light mt-2 rounded-top border-top border-left border-right',
    ),
    dbc.Row(
        [
            dbc.Col(
                html.H5(
                    '{:.2f} hours'.format(num_hours),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5(len(data), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5(
                    '{} words'.format(len(vocabulary)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5(
                    '{} chars'.format(len(alphabet)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
            ),
        ],
        className='bg-light rounded-bottom border-bottom border-left border-right',
    ),
    dbc.Row(dbc.Col(html.H5(children='Alphabet'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        dbc.Col(html.Div('{}'.format(sorted(alphabet))),), className='mt-2 bg-light text-monospace rounded border'
    ),
    dbc.Row(dbc.Col(html.H5('Duration (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='duration-graph', figure=figure_duration),),),
    dbc.Row(dbc.Col(html.H5('Number of words (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='num-words-graph', figure=figure_num_words),),),
    dbc.Row(dbc.Col(html.H5('Number of characters (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='num-chars-graph', figure=figure_num_chars),),),
    dbc.Row(dbc.Col(html.H5('Word rate (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='word-rate-graph', figure=figure_word_rate),),),
    dbc.Row(dbc.Col(html.H5('Character rate (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='char-rate-graph', figure=figure_char_rate),),),
    dbc.Row(dbc.Col(html.H5('Vocabulary'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='wordstable',
                columns=[{'name': 'Word', 'id': 'word'}, {'name': 'Count', 'id': 'count'}],
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                page_count=math.ceil(len(vocabulary) / DATA_PAGE_SIZE),
                sort_by=[{'column_id': 'word', 'direction': 'asc'}],
                style_cell={'maxWidth': 0, 'textAlign': 'left'},
                style_header={'color': 'text-primary'},
            ),
        ),
        className='m-2',
    ),
]


@app.callback(
    [Output('wordstable', 'data'), Output('wordstable', 'page_count')],
    [Input('wordstable', 'page_current'), Input('wordstable', 'sort_by'), Input('wordstable', 'filter_query')],
)
def update_wordstable(page_current, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
            print(len(vocabulary_view), len(vocabulary))
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]
            print(len(vocabulary_view), len(vocabulary))

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(vocabulary_view):
        page_current = len(vocabulary_view) // DATA_PAGE_SIZE
    return [
        vocabulary_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(vocabulary_view) / DATA_PAGE_SIZE),
    ]


samples_layout = (
    [
        dbc.Row(dbc.Col(html.H5('Data'), className='text-secondary'), className='mt-3'),
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id='datatable',
                    columns=[{'name': k.replace('_', ' '), 'id': k} for k in data[0]],
                    filter_action='custom',
                    filter_query='',
                    sort_action='custom',
                    sort_mode='single',
                    sort_by=[],
                    row_selectable='single',
                    selected_rows=[0],
                    page_action='custom',
                    page_current=0,
                    page_size=DATA_PAGE_SIZE,
                    page_count=math.ceil(len(data) / DATA_PAGE_SIZE),
                    style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'left'},
                    style_header={'color': 'text-primary', 'text_align': 'center',},
                    style_cell_conditional=[{'if': {'column_id': 'audio_filepath'}, 'width': '15%'}]
                    + [
                        {'if': {'column_id': c}, 'width': '10%', 'text_align': 'center'}
                        for c in ['duration', 'num_words', 'num_chars', 'word_rate', 'char_rate']
                    ],
                ),
            )
        ),
    ]
    + [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(children=k.replace('_', ' ')),
                    width=2,
                    className='mt-1 bg-light text-monospace text-break small rounded border',
                ),
                dbc.Col(
                    html.Div(id='_' + k), className='mt-1 bg-light text-monospace text-break small rounded border'
                ),
            ]
        )
        for k in data[0]
    ]
    + [
        dbc.Row(dbc.Col(html.Audio(id='player', controls=True),), className='mt-3 '),
        dbc.Row(dbc.Col(dcc.Graph(id='signal-graph')), className='mt-3'),
    ]
)


@app.callback(
    [Output('datatable', 'data'), Output('datatable', 'page_count')],
    [Input('datatable', 'page_current'), Input('datatable', 'sort_by'), Input('datatable', 'filter_query')],
)
def update_datatable(page_current, sort_by, filter_query):
    data_view = data
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            data_view = [x for x in data_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            data_view = [x for x in data_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        data_view = sorted(data_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(data_view):
        page_current = len(data_view) // DATA_PAGE_SIZE
    return [
        data_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(data_view) / DATA_PAGE_SIZE),
    ]


app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
            ],
            brand='Speech Data Explorer',
            sticky='top',
            color='green',
            dark=True,
        ),
        dbc.Container(id='page-content'),
    ]
)


@app.callback(
    [Output('page-content', 'children'), Output('stats_link', 'active'), Output('samples_link', 'active')],
    [Input('url', 'pathname')],
)
def nav_click(url):
    if url == '/samples':
        return [samples_layout, False, True]
    else:
        return [stats_layout, True, False]


@app.callback(
    [Output('_' + k, 'children') for k in data[0]], [Input('datatable', 'selected_rows'), Input('datatable', 'data')]
)
def show_item(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    return [data[idx[0]][k] for k in data[0]]


@app.callback(Output('signal-graph', 'figure'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        filename = data[idx[0]]['audio_filepath']
        audio, fs = librosa.load(filename, sr=None)
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[[0, 'rgb(30,62,62)'], [0.5, 'rgb(30,128,128)'], [1, 'rgb(30,255,30)'],],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception:
        pass

    return figs


@app.callback(Output('player', 'src'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        filename = data[idx[0]]['audio_filepath']
        encoded = base64.b64encode(open(filename, 'rb').read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception:
        return ''


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=args.port, debug=args.debug)
