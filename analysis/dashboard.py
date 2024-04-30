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
experiments, outcomes = load_results('../results/0409_100_runs.tar.gz')

# Convert outcomes from ReplicatorModel-style to Model-style
new_outcomes = {}
for var in outcomes:
    new_outcomes[var] = []
    for a in outcomes[var]:
        new_outcomes[var].append(list(a[0]))
    outcomes[var] = np.array(new_outcomes[var])

N_RUNS = len(outcomes['GDP'])
N_STEPS = len(outcomes['GDP'][0])
N_OUTCOMES = len(outcomes)

# Combine data into one DF
all_data = {}

#  --> Experiment columns
all_data['Run'] = []
for input in experiments:
    all_data[input] = []
all_data['Step'] = []
for i in range(N_RUNS):
    for j in range(N_STEPS):
        all_data['Run'].append(i)
        for input in experiments:
            all_data[input].append(experiments.loc[i, input])
        all_data['Step'].append(j)
#  --> Data columns
for outcome, data in outcomes.items():
    if outcome == 'TIME':
        continue
    all_data[outcome] = []
    # print(data)
    for run in data:
        all_data[outcome] += [val for val in run]
#  --> Make DF
DATA = pd.DataFrame(all_data)

# Get inputs (parameters) and outputs (KPIs)
input_names = sorted(experiments.columns.tolist())
outcome_names = [k for k in outcomes]
# Ignore EMA columns
# TODO: For now, also ignore flood_narrative bc it's handled by group_by. Can put in later
for col in ['scenario', 'policy', 'model', 'flood_narrative']:
    input_names.remove(col)

# Get bounds for each input
INPUT_MIN = {
    'debt_sales_ratio' : 0.8,
    'init_markup' : 0.05,
    'wage_sensitivity_prod' : 0.0
    # input : experiments[input].min()
    # for input in input_names
}
INPUT_MAX = {
    'debt_sales_ratio' : 5.0,
    'init_markup' : 0.5,
    'wage_sensitivity_prod' : 1.0
    # input : experiments[input].max()
    # for input in input_names
}

# Get flood types and associate with colours
COLORS = px.colors.qualitative.Plotly
FLOODS = experiments['flood_narrative'].unique()
FLOOD_COLOR_MAP = {flood : COLORS[i] for i, flood in enumerate(FLOODS)}
FLOOD_NAME_MAP = {flood : 'ABCDEFGHIJK'[i] for i, flood in enumerate(FLOODS)}
FLOOD_DASH_MAP = {1000:'solid', 100:'dash', 10:'dot'}

# Clean up
del all_data
del experiments
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
        ] + [
            html.P(
                className='header-sub',
                children=f"{FLOOD_NAME_MAP[flood_string] + ': ' + flood_string}"
            ) for flood_string in FLOODS
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
                            html.H3(className='subtitle', children='Flood Handling'),
                            dcc.RadioItems(options=['Combine', 'Group By', 'Separate'],
                                        value='Combine',
                                        id='flood-handling',
                                        inline=True),
                        ]
                    ),
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
    Input('flood-handling', 'value'),
    Input('show-legend', 'value'),
    Input('show-inputs', 'value'),
    Input({'type' : 'input-bounds-store',
           'index': ALL}, 'data'),
    Input('highlighting-mode', 'value'),
    Input({'type' : 'input-highlight-range-store',
           'index': ALL}, 'data'),
    Input('color-param', 'value'),
)
def update_graph(outcomes, flood_handling, show_legend, show_inputs, bounds_stores,
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
        if flood_handling == 'Separate':
            # Creating one plot per (outcome, flood type) pair
            for flood_string in FLOODS:
                # Create base graph with traces
                fig = create_graph(
                    DATA[DATA.flood_narrative == flood_string],
                    outcome,
                    title=f'{outcome} ({FLOOD_NAME_MAP[flood_string]})',
                    show_legend=show_legend,
                    show_inputs=show_inputs,
                    input_bounds=bounds_stores,
                    highlight_mode=highlight_mode,
                    input_highlight_ranges=highlight_range_stores,
                    color_param=color_param,
                )

                # Add vertical lines for each flood
                flood_dict = {
                    int(flood.split(':')[0].strip()) : int(flood.split(':')[1].strip())
                    for flood in flood_string[1:-1].split(',')
                }
                for flood_time, flood_return in flood_dict.items():
                    line_dash = FLOOD_DASH_MAP[flood_return]
                    fig.add_vline(x=flood_time,
                                  line_width=2,
                                  line_dash=line_dash,
                                  line_color="red")
                
                # Add graph to body
                graphs.append(html.Div(
                    children=dcc.Graph(
                        id=f'{flood_string}-{outcome}-graph',
                        figure=fig
                    ),
                    className='card'
                ))
        else:
            # Creating one plot for whole outcome
            group_by = (flood_handling == 'Group By')

            # Create graph from traces
            fig = create_graph(
                DATA,
                outcome,
                group_by=group_by,
                show_legend=show_legend,
                show_inputs=show_inputs,
                input_bounds=bounds_stores,
                highlight_mode=highlight_mode,
                input_highlight_ranges=highlight_range_stores,
                color_param=color_param,
            )
            # Add graph to body
            graphs.append(html.Div(
                children=dcc.Graph(
                    id=f'{outcome}-graph',
                    figure=fig
                ),
                className='card'
            ))
    return graphs

def create_graph(plot_data, outcome, group_by=False,
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
    first_run_handled = {flood:False for flood in FLOODS}
    for run in range(N_RUNS):
        # Filter data to specific run
        run_data = plot_data[plot_data.Run == run]

        # Skip run if out of bounds
        skip = False
        for i, input in enumerate(input_names):
            # Extract bounds for this input
            bounds = input_bounds[i].split(':')

            # Extract experiment level for this input
            level = run_data[input].max() 

            # Compare (level OUTSIDE bounds ==> skip)
            if level < float(bounds[0]) or level > float(bounds[1]):
                skip = True
        if skip:
            continue

        # Handle adding traces based on flood_handling
        # :: if 'Group By', need to colour categorically
        if group_by:
            # Identify current run's flood type
            flood = run_data['flood_narrative'].tolist()[0]
            # Add trace to figure, coloured according to group, and only
            #  show the legend if it's the first run of its flood type
            fig.add_trace(go.Scatter(
                x=run_data.Step,
                y=run_data[outcome],
                line=dict(color=FLOOD_COLOR_MAP[flood], width=1),
                name=f'Run {run}',
                meta={input : run_data[input].max() for input in input_names},
                hovertemplate=hover,
                legendgroup=flood,
                legendgrouptitle_text=FLOOD_NAME_MAP[flood],  
                showlegend=(show_legend or not first_run_handled[flood]),
            ))
            # Update progress for legend handling
            if not first_run_handled[flood]:
                first_run_handled[flood] = True
        # :: else, draw all lines as grey (TODO: add highlighting rules here)
        else: 
            # Determine line color
            if highlight_mode == 'Ranges':
                # Assume highlighting. If input outside the range, set to False
                highlight = True
                # If all ranges are same as input bounds, don't highlight
                if input_highlight_ranges == input_bounds:
                    highlight = False
                else:
                    for i, input in enumerate(input_names):
                        # Extract highlight_range for this input
                        highlight_range = input_highlight_ranges[i].split(':')
                        # Extract experiment level for this input
                        level = run_data[input].max() 
                        # Compare (level OUTSIDE highlight range ==> don't highlight)
                        if level < float(highlight_range[0]) or level > float(highlight_range[1]):
                            highlight = False
                # Determine line colour 
                color = 'rgba(255,102,146,0.75)' if highlight else 'rgba(128,128,128,0.5)'
            else:
                if color_param is None:
                    color = 'rgba(128,128,128,0.5)'
                else: 
                    min = INPUT_MIN[color_param]
                    max = INPUT_MAX[color_param]
                    level = run_data[color_param].max()
                    fraction = (max - level) / (max - min)
                    color = px.colors.sample_colorscale('viridis', fraction)[0]
            # Create hover tooltip
            hover = '%{y:.4f} @ t=%{x}'
            if show_inputs:
                hover += ''.join(['<br>'+i+': %{meta.'+i+':0.4f}' for i in input_names])

            # Add trace to figure
            fig.add_trace(go.Scatter(
                x=run_data.Step,
                y=run_data[outcome],
                line=dict(color=color, width=1),
                name=f'Run {run}',
                meta={input : run_data[input].max() for input in input_names},
                hovertemplate=hover,
                showlegend=show_legend,
            ))
    return fig
    # return px.line(outcomes_df, x='Step', y=outcome, line_group='Run', color='Run')

if __name__ == '__main__':
    app.run(debug=True)