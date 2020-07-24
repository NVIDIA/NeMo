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
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import librosa
import numpy as np
from dash.dependencies import Input, Output
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


def parse_args():
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    return args.manifest, args.debug


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
            num_hours += item['duration']
            for word in item['text'].split():
                vocabulary[word] += 1
            for char in item['text']:
                alphabet.add(char)
    num_hours /= 60.0 * 60.0
    return data, num_hours, vocabulary, alphabet


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


manifest_file, debug = parse_args()
data, num_hours, vocabulary, alphabet = load_data(manifest_file)

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
                data=[{'word': word, 'count': vocabulary[word]} for word in vocabulary],
                filter_action='native',
                sort_action='native',
                sort_by=[{'column_id': 'word', 'direction': 'asc'}],
                page_current=0,
                page_size=10,
                style_cell={'maxWidth': 0, 'textAlign': 'left'},
                style_header={'color': 'text-primary'},
            ),
        ),
        className='m-2',
    ),
]

samples_layout = [
    dbc.Row(dbc.Col(html.H5('Data'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='datatable',
                columns=[{'name': k.replace('_', ' '), 'id': k} for k in data[0]],
                data=data,
                filter_action='native',
                sort_action='native',
                row_selectable='single',
                selected_rows=[0],
                page_action='native',
                page_current=0,
                page_size=10,
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
    dbc.Row(dbc.Col(html.Div(id='filename'),), className='mt-2 bg-light text-monospace text-break rounded border'),
    dbc.Row(dbc.Col(html.Div(id='transcript'),), className='mt-2 bg-light text-monospace rounded border'),
    dbc.Row(dbc.Col(html.Audio(id='player', controls=True),), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='signal-graph')), className='mt-3'),
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
    [Output('filename', 'children'), Output('transcript', 'children')], [Input('datatable', 'selected_rows')]
)
def show_text(idx):
    text = data[idx[0]]['text']
    return data[idx[0]]['audio_filepath'], text


@app.callback(Output('signal-graph', 'figure'), [Input('datatable', 'selected_rows')])
def plot_signal(idx):
    filename = data[idx[0]]['audio_filepath']
    audio, fs = librosa.load(filename, sr=None)
    time_stride = 0.01
    hop_length = int(fs * time_stride)
    n_fft = 512
    # linear scale spectrogram
    s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    figs.add_trace(
        go.Scatter(x=np.arange(audio.shape[0]) / fs, y=audio, line={'color': 'green'}, name='Waveform'), row=1, col=1
    )
    figs.add_trace(
        go.Heatmap(
            z=s_db,
            colorscale=[[0, 'rgb(30,62,62)'], [0.5, 'rgb(30,128,128)'], [1, 'rgb(30,255,30)'],],
            colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
            dx=time_stride,
            dy=fs / n_fft / 1000,
            name='Spectrogram',
        ),
        row=2,
        col=1,
    )
    figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
    figs.update_xaxes(title_text='Time, s', row=1, col=1)
    figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    figs.update_xaxes(title_text='Time, s', row=2, col=1)

    return figs


@app.callback(Output('player', 'src'), [Input('datatable', 'selected_rows')])
def update_player(idx):
    filename = data[idx[0]]['audio_filepath']
    encoded = base64.b64encode(open(filename, 'rb').read())
    return 'data:audio/wav;base64,{}'.format(encoded.decode())


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=debug)
