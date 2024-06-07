from dash import Dash, html, dcc, callback, Output, Input, State, MATCH, ALL, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json 

from ema_workbench import (
    load_results
)

# Load data
experiments, outcomes = load_results('../results/1k_run_0604.tar.gz')

# Convert outcomes from ReplicatorModel-style to Model-style
# new_outcomes = {}
# for var in outcomes:
#     new_outcomes[var] = []
#     for a in outcomes[var]:
#         new_outcomes[var].append(list(a[0]))
#     outcomes[var] = np.array(new_outcomes[var])

N_RUNS = len(outcomes['GDP'])
N_STEPS = len(outcomes['GDP'][0])
N_OUTCOMES = len(outcomes)


# Get inputs (parameters) and outputs (KPIs)
input_names = sorted(experiments.columns.tolist())
outcome_names = [k for k in outcomes]
# Ignore EMA columns
for col in ['scenario', 'policy', 'model']:
    input_names.remove(col)

# Get bounds for each input
INPUT_MIN = {}
INPUT_MAX = {}
for col in input_names:
    min_val = experiments[col].min()
    max_val = experiments[col].max()
    if experiments[col].dtype == np.int64:
        INPUT_MIN[col] = min_val
        INPUT_MAX[col] = max_val
    else:
        INPUT_MIN[col] = ((min_val*10**2)//1)/(10**2) # floor with precision
        INPUT_MAX[col] = ((1+max_val*10**2)//1)/(10**2)

# Create dataframes for each outcome
outcome_dfs = {}
for outcome in outcomes:
    outcome_dfs[outcome] = pd.DataFrame(outcomes[outcome])
del outcomes

# Create Viridis swatch
VIRIDIS_FIG = px.colors.sequential.swatches_continuous()
VIRIDIS_FIG.data = [VIRIDIS_FIG.data[64]]
VIRIDIS_FIG.update_layout({
    'width' : 280,
    'height' : 100,
    'title' : {
        'text':'Scale',
        'pad' : {'b':5,'l':0,'r':0,'t':5}
    },
    'margin' : {'b':10,'l':10,'r':10,'t':40},
})

app = Dash(__name__)

app.layout = html.Div([
    html.Div(
        className='header',
        children=[
            html.H1(
                className='header-title',
                children='CRAB Exploratory Modeling Dashboard',
            )
        ]
    ),
    html.Div(
        className='row',
        children=[
            html.Div(
                className='col side', 
                children=[
                    html.Div(
                        children=[
                            html.H3(className='subtitle', children='Outcomes'),
                            dcc.Checklist(options=outcome_names,
                                        id='outcome-selector')
                        ]
                    )
                ]
            ),
            html.Div(
                className='col middle wrapper',
                id='graph-content',
            ),
            html.Div(
                className='col side',
                children=[
                    html.Div(
                        children=[
                            html.H3(className='subtitle', children='Plotting Options'),
                            html.H5(className='subsubtitle', children='Plot Legends'),
                            dcc.RadioItems(options=['Show', 'Hide'],
                                           value='Hide',
                                           id='show-legend',
                                           inline=True),
                            html.H5(className='subsubtitle', children='Inputs in Hover Text'),
                            dcc.RadioItems(options=['Show', 'Hide'],
                                           value='Show',
                                           id='show-inputs',
                                           inline=True),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H3(className='subtitle', children='Parameter Bounding'),
                        ] + [
                            html.Div(
                                children=[
                                    html.H5(
                                        className='subsubtitle',
                                        children=f'{input}'
                                    ),
                                    dcc.RangeSlider(
                                        min=INPUT_MIN[input],
                                        max=INPUT_MAX[input],
                                        value=[INPUT_MIN[input], INPUT_MAX[input]],
                                        marks=None,
                                        updatemode='drag',
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                            "style": {
                                                "color": "SteelBlue",
                                                "fontSize": "12px",
                                            },
                                        },
                                        id={'type' : 'input-bounds-slider',
                                            'index': input}
                                    ),
                                ]
                            )
                            for input in input_names
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H3(className='subtitle', children='Parameter Highlighting'),
                            dcc.RadioItems(options=['Ranges', 'Single Parameter'],
                                        value='Ranges',
                                        id='highlighting-mode',
                                        inline=True),
                            html.Div(
                                children=[
                                    dcc.Dropdown(options=input_names,
                                                 id='color-param'),
                                    dcc.Graph(figure=VIRIDIS_FIG,
                                              id='color-swatch'),
                                ],
                                hidden=True,
                                id='color-param-wrapper',
                            ), 
                            html.Div(
                                children=[
                                    html.Div(
                                        children=[
                                            html.H5(
                                                className='subsubtitle',
                                                children=f'{input}'
                                            ),
                                            dcc.RangeSlider(
                                                min=INPUT_MIN[input],
                                                max=INPUT_MAX[input],
                                                value=[INPUT_MIN[input], INPUT_MAX[input]],
                                                marks=None,
                                                updatemode='drag',
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                    "style": {
                                                        "color": "SteelBlue",
                                                        "fontSize": "12px",
                                                    },
                                                },
                                                id={'type' : 'input-highlight-range-slider',
                                                    'index': input}
                                            ),
                                        ]
                                    )
                                    for input in input_names
                                ],
                                id='param-ranges-wrapper',
                            )
                        ]
                    ),
                ]
            )
        ]
    ),
    html.Div(
        children=[
            dcc.Store(
                id={'type':'input-bounds-store', 'index':input},
                data='',
            )
            for input in input_names
        ] + [
            dcc.Store(
                id={'type':'input-highlight-range-store', 'index':input},
                data='',
            )
            for input in input_names
        ]
    )
])

@callback(
    Output('color-param-wrapper', 'hidden'),
    Output('param-ranges-wrapper', 'hidden'),
    Input('highlighting-mode', 'value'),
)
def toggle_highlighting_mode(mode):
    if mode == 'Single Parameter':
        return False, True
    else:
        return True, False

@callback(
    Output({'type' : 'input-bounds-store',
           'index': MATCH}, 'data'),
    Output({'type' : 'input-highlight-range-store',
           'index': MATCH}, 'data'),
    Output({'type' : 'input-highlight-range-slider',
           'index': MATCH}, 'value'),
    Input({'type' : 'input-bounds-slider',
           'index': MATCH}, 'value'),
    Input({'type' : 'input-highlight-range-slider',
           'index': MATCH}, 'value'),
)
def update_highlight_sliders(new_bounds, new_highlight_range):
    # Bound highlight range by parameter bound
    bounded_highlight_range = new_highlight_range
    if new_bounds[0] > new_highlight_range[0]:
        bounded_highlight_range[0] = new_bounds[0]
    if new_bounds[1] < new_highlight_range[1]:
        bounded_highlight_range[1] = new_bounds[1]
    
    # Store new bounds & highlight range
    new_bounds_str = f'{new_bounds[0]}:{new_bounds[1]}'
    new_highlight_range_str = f'{bounded_highlight_range[0]}:{bounded_highlight_range[1]}'

    return new_bounds_str, new_highlight_range_str, new_highlight_range
    

@callback(
    Output('graph-content', 'children'),
    Input('outcome-selector', 'value'),
    Input('show-legend', 'value'),
    Input('show-inputs', 'value'),
    Input({'type' : 'input-bounds-store',
           'index': ALL}, 'data'),
    Input('highlighting-mode', 'value'),
    Input({'type' : 'input-highlight-range-store',
           'index': ALL}, 'data'),
    Input('color-param', 'value'),
)
def update_graph(outcomes,
                 show_legend, show_inputs, bounds_stores,
                 highlight_mode, highlight_range_stores, color_param):
    # Quit early if no outcomes selected    
    if outcomes is None:
        return []
    
    # Interpret legend settings
    show_legend = True if show_legend=='Show' else False
    show_inputs = True if show_inputs=='Show' else False

    # Create appropriate graphs
    graphs = []
    for outcome in outcomes:
        # Create graph from traces
        fig = create_graph(
            outcome,
            # group_by=group_by,
            show_legend=show_legend,
            show_inputs=show_inputs,
            input_bounds=bounds_stores,
            highlight_mode=highlight_mode,
            input_highlight_ranges=highlight_range_stores,
            color_param=color_param,
        )
        # fig.write_image(f'../results/{outcome}.svg', scale=1.0, width=800, height=400)
        # Add graph to body
        graphs.append(html.Div(
            children=dcc.Graph(
                id=f'{outcome}-graph',
                figure=fig
            ),
            className='card'
        ))
    return graphs

def create_graph(outcome, group_by=False,
                 title=None, show_legend=False, show_inputs=False,
                 input_bounds=None, highlight_mode='Ranges',
                 input_highlight_ranges=None, color_param=None):
    # Create figure
    if title is None:
        title = outcome
    fig = go.Figure(layout={
        'title' : go.layout.Title(text=title),
    })
    
    # Plot lines
    for run in range(N_RUNS):
        # Select outcome data
        run_data = outcome_dfs[outcome].iloc[run, :]

        # Skip run if out of bounds
        skip = False
        for i, input in enumerate(input_names):
            # Extract bounds for this input
            bounds = input_bounds[i].split(':')
            # Extract experiment level for this input
            level = experiments.loc[run, input] 
            # Compare (level OUTSIDE bounds ==> skip)
            if level < float(bounds[0]) or level > float(bounds[1]):
                skip = True
                break
        if skip:
            continue

        # Determine line color
        if highlight_mode == 'Ranges':
            highlight = True
            # If all ranges are same as input bounds, don't highlight
            if input_highlight_ranges == input_bounds:
                highlight = False
            else:
                # Otherwise, if input outside the range, also don't highlight
                for i, input in enumerate(input_names):
                    # Extract highlight_range for this input
                    highlight_range = input_highlight_ranges[i].split(':')
                    # Extract experiment level for this input
                    level = experiments.loc[run, input]
                    # Compare (level OUTSIDE highlight range ==> don't highlight)
                    if level < float(highlight_range[0]) or level > float(highlight_range[1]):
                        highlight = False
            color = 'rgba(255,102,146,0.325)' if highlight else 'rgba(128,128,128,0.15)'
        elif highlight_mode == 'Parameter':
            if color_param is None:
                color = 'rgba(128,128,128,0.5)'
            else: 
                min = INPUT_MIN[color_param]
                max = INPUT_MAX[color_param]
                level = experiments.loc[run, color_param]
                fraction = (max - level) / (max - min)
                color = px.colors.sample_colorscale('viridis', fraction)[0]

        # Create hover tooltip to display input levels when hovering over a line
        meta = {input : experiments.loc[run, input] for input in input_names}
        hover = '%{y:.4f} @ t=%{x}'
        if show_inputs:
            hover += ''.join(['<br>'+i+': %{meta.'+i+':0.4f}' for i in input_names])

        # Add trace to figure
        fig.add_trace(go.Scatter(
            x=run_data.index,
            y=run_data.values,
            line=dict(color=color, width=1),
            name=f'Run {run}',
            meta=meta,
            hovertemplate=hover,
            showlegend=show_legend,
        ))
    return fig

if __name__ == '__main__':
    app.run(debug=True)