from dash import Dash, html, dcc, callback, Output, Input, State, Patch, MATCH, ALL, ctx
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pxc
import pandas as pd
import numpy as np
import json 

from ema_workbench import (
    load_results
)

# Load data
# experiments, outcomes = load_results('../results/10k_run_0605__with_clusters.tar.gz')
experiments, outcomes = load_results('../results/1k_run_0604__with_clusters.tar.gz')
# experiments, outcomes = load_results('../results/stochasticity_test_0607.tar.gz')

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

# Categorical colours for cluster grouping
HEX_COLORS = px.colors.qualitative.G10
RGB_COLORS = []
for color in HEX_COLORS:
    r,g,b = pxc.hex_to_rgb(color)
    RGB_COLORS.append(f'rgba({r}, {g}, {b}, 1)')
PINK = 'rgba(247, 113, 137, 0.3)'
ORANGE = 'rgba(213, 140, 50, 0.4)'
GREY = 'rgba(128, 128, 128, 0.15)'

# Get inputs (parameters) and outputs (KPIs)
input_names = sorted(experiments.columns.tolist())
outcome_names = [k for k in outcomes]

# Determine which outcomes were used for clustering:
clustered_outputs = []
for col in input_names: 
    if 'Cluster' in col:
        clustered_outputs.append(col)

# Ignore EMA columns and cluster columns
for col in ['scenario', 'policy', 'model'] + clustered_outputs:
    input_names.remove(col)
if 'seed' in input_names:
    input_names.remove('seed')

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

# Map cluster : z-order by cluster size:
CLUSTER_Z_MAP = {}
for cluster_type in clustered_outputs:
    CLUSTER_Z_MAP[cluster_type] = {}

    cluster_counts = experiments[cluster_type].value_counts()
    cluster_counts = cluster_counts.sort_values(ascending=False)

    for z, cluster in enumerate(cluster_counts.index):
        CLUSTER_Z_MAP[cluster_type][cluster] = z*10

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
                            dcc.RadioItems(options=['Ranges', 'Parameter', 'Clusters'],
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
                                    dcc.Dropdown(options=clustered_outputs,
                                                 id='cluster-var'),
                                    dcc.RadioItems(
                                        options=['All'],
                                        value='All',
                                        id='cluster-val',
                                        inline=True
                                    )
                                ],
                                hidden=True,
                                id='cluster-var-wrapper',
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
    Output('cluster-var-wrapper', 'hidden'),
    Output('param-ranges-wrapper', 'hidden'),
    Input('highlighting-mode', 'value'),
)
def toggle_highlighting_mode(mode):
    if mode == 'Parameter':
        return False, True, True
    if mode == 'Clusters':
        return True, False, True
    else:
        return True, True, False

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
    Output('cluster-val', 'value'),
    Output('cluster-val', 'options'),
    Input('cluster-var', 'value'),
)
def update_cluster_options(cluster_var):
    if cluster_var is None:
        return ('All', ['All'])
    else:
        num_clusters = experiments.nunique()[cluster_var]
        return ('All', ['All'] + list(range(num_clusters)))
    
@callback(
    Output('graph-content', 'children'),
    Input('outcome-selector', 'value'),
    Input('show-legend', 'value'),
    Input('show-inputs', 'value'),
    Input('highlighting-mode', 'value'),
    Input('color-param', 'value'),
    Input('cluster-var', 'value'),
    Input('cluster-val', 'value'),
    Input({'type' : 'input-bounds-store',
           'index': ALL}, 'data'),
    Input({'type' : 'input-highlight-range-store',
           'index': ALL}, 'data'),
)
def create_graphs(outcomes,
                 show_legend, show_inputs, highlight_mode,
                 color_param, cluster_var, cluster_val,
                 bounds_stores, highlight_range_stores):
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
        fig = create_single_graph(
            outcome,
            show_legend=show_legend,
            show_inputs=show_inputs,
            highlight_mode=highlight_mode,
            color_param=color_param,
            cluster_var=cluster_var,
            cluster_val=cluster_val,
            input_bounds=bounds_stores,
            input_highlight_ranges=highlight_range_stores,
        )
        # fig.write_image(f'../results/{outcome}.svg', scale=1.0, width=800, height=400)
        # Add graph to body
        graphs.append(html.Div(
            children=dcc.Graph(
                id={'type':'graph','index':outcome},
                figure=fig
            ),
            className='card'
        ))
    return graphs

def create_single_graph(outcome, title=None, show_legend=False, show_inputs=False,
                        highlight_mode='Ranges', color_param=None,
                        cluster_var=None, cluster_val=None,
                        input_bounds=None, input_highlight_ranges=None):
    # Create figure
    if title is None:
        title = outcome
    fig = go.Figure(layout={
        'title' : go.layout.Title(text=title),
    })
    
    # Plot traces
    traces = {}
    cluster_legendgroup_handled = {}
    for run in range(N_RUNS):
        # Select outcome data
        run_data = outcome_dfs[outcome].iloc[run, :]

        ##########################
        ### PARAMETER BOUNDING ###
        ##########################

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

        ###############################
        ### LINE STYLE & Z-ORDERING ###
        ###############################

        linestyle = dict(color=GREY, width=1)
        z = 0

        # Colour lines based on selected 'Highlighting Mode'
        if highlight_mode == 'Ranges':
            # Assume highlighting
            linestyle['color'] = PINK
            z = 10
            # If all ranges are same as input bounds, don't highlight
            if input_highlight_ranges == input_bounds:
                linestyle['color'] = GREY
                z = 0
            else:
                # Otherwise, if input outside the range, also don't highlight
                for i, input in enumerate(input_names):
                    # Extract highlight_range for this input
                    highlight_range = input_highlight_ranges[i].split(':')
                    # Extract experiment level for this input
                    level = experiments.loc[run, input]
                    # Compare (level OUTSIDE highlight range ==> don't highlight)
                    if level < float(highlight_range[0]) or level > float(highlight_range[1]):
                        linestyle['color'] = GREY
                        z = 0
                        break

        elif highlight_mode == 'Parameter':
            if color_param is not None:
                min = INPUT_MIN[color_param]
                max = INPUT_MAX[color_param]
                level = experiments.loc[run, color_param]
                fraction = (max - level) / (max - min)
                linestyle['color'] = px.colors.sample_colorscale('viridis', fraction)[0]

        elif highlight_mode == 'Clusters':
            if cluster_var is not None:
                if cluster_val is None or cluster_val == 'All':
                    cluster = experiments.loc[run, cluster_var]
                    linestyle['color'] = RGB_COLORS[cluster]
                    z = CLUSTER_Z_MAP[cluster_var][cluster]
                else: 
                    cluster = experiments.loc[run, cluster_var]
                    if cluster == cluster_val:
                        # TODO: Add Z-ordering: main-layer and top-layer, add traces after loop.
                        linestyle['color'] = ORANGE
                        z = 10

        #####################
        ### LINE METADATA ###
        #####################

        # Create hover tooltip to display input levels when hovering over a line
        meta = {input : experiments.loc[run, input] for input in input_names}
        hover = '%{y:.4f} @ t=%{x}'
        if show_inputs:
            hover += ''.join(['<br>'+i+': %{meta.'+i+':0.4f}' for i in input_names])

        # Handle presenting seeds in tooltip, for a seeded run
        if 'seed' in experiments:
            meta['seed'] = experiments.loc[run, 'seed']
            hover += '<br>seed: %{meta.seed}'

        # Decide whether to show legend for this trace
        show_legend_trace = show_legend

        # If grouping by clusters, create a group of traces in the plot's legend
        legendgroup = None
        legendgrouptitle = None
        if highlight_mode == 'Clusters' and cluster_var is not None:
            # Read cluster value
            cluster = experiments.loc[run, cluster_var]
            # Extend tooltip info
            meta['cluster'] = cluster
            if show_inputs:
                hover += '<br>'+cluster_var.lower()+': %{meta.cluster}'
            # Legend group == cluster
            legendgroup = str(cluster)
            # Give group descriptive name
            legendgrouptitle = f"{cluster_var}: {cluster}"
            # Only show legend for 1st run of a cluster
            if cluster not in cluster_legendgroup_handled:
                cluster_legendgroup_handled[cluster] = True
            else:
                show_legend_trace = False

        if z in traces:
            traces[z].append(go.Scatter(
                x=run_data.index,
                y=run_data.values,
                line=linestyle,
                name=f'Run {run}',
                meta=meta,
                hovertemplate=hover,
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgrouptitle,  
                showlegend=show_legend_trace,
            ))
        else:
            traces[z] = [go.Scatter(
                x=run_data.index,
                y=run_data.values,
                line=linestyle,
                name=f'Run {run}',
                meta=meta,
                hovertemplate=hover,
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgrouptitle,  
                showlegend=show_legend_trace,
            )]
    
    # Add traces to figure in z-order
    z_vals = sorted(traces.keys())
    for z in z_vals:
        for trace in traces[z]:
            fig.add_trace(trace)

    return fig

if __name__ == '__main__':
    app.run(debug=True)