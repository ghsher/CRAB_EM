from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ema_workbench.analysis import lines

def plot_timeseries(data, floods=None):
    """
    Plotting a single timeseries from CRAB model data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the time series data.
    floods : list, optional
        A list of flood times to be represented as vertical bars on the plot.

    Returns:
    --------
    None
    """
    # Setup
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))
    
    # Add vertical bars to represent floods
    # TODO: Add flood_intensities argument to inform line style
    if floods is not None:
        for flood_time in floods:
            plt.axvline(x=flood_time, color='r', linestyle='--', linewidth=1)
    
    # Plot the timeseries
    sns.lineplot(data=data, linewidth=2)

    # Tweak visuals
    sns.despine()

    plt.show()

def plot_timeseries_subplots(data, plt_height=5, title=''):
    """
    Plots >1 timeseries from any number of runs in one figure.
    REQUIREMENT: The runs must have the same flooding pattern.

    Parameters:
    -----------
    data : pandas.DataFrame or list(pandas.DataFrame)
        The DataFrame containing the time series data.
    plt_height : int, optional
        Height of each plot for matplotlib. Defaults to 5
    title : str, optional
        Title of entire figure (all subplots together)

    Returns:
    --------
    None
    """
    ## TYPE HANDLING ##
    if not isinstance(data, list):
        data = [data]
    
    ## FLOOD EXTRACTION ##
    flood_ticks = []
    flood_intensities = []
    cols = data[0].columns
    if 'Flood' in cols:
        flood_ticks = data[0][data[0]['Flood'] == True].index
        if 'Flood intensity' in data[0].columns:
            flood_intensities = data[0][data[0]['Flood'] == True]['Flood intensity']
    cols = [col for col in cols if col != 'Flood' and col != 'Flood intensity']
    # TODO: Assert on whether the "floods" & "flood intensities" are identical
    n_plots = len(cols)

    # Setup
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, plt_height*n_plots), ncols=1, nrows=n_plots)
    
    if n_plots == 1:
        ax = [ax]
    for plot_id, col in enumerate(cols):
        plot_data = pd.DataFrame()
        for run_id, df in enumerate(data):
            plot_data[run_id] = df[col]
        # Add vertical bars to represent floods
        # TODO: Add flood_intensities argument to inform line style
        #       (requires model to report flood intensity as an output alongside flood)
        if len(flood_ticks) > 0:
            for flood, flood_tick in enumerate(flood_ticks):
                if flood_intensities.iloc[flood] == 10:
                    ax[plot_id].axvline(x=flood_tick, color='r', linestyle=':', linewidth=2)
                elif flood_intensities.iloc[flood] == 100:
                    ax[plot_id].axvline(x=flood_tick, color='r', linestyle='--', linewidth=2)
                elif flood_intensities.iloc[flood] == 1000:
                    ax[plot_id].axvline(x=flood_tick, color='r', linestyle='-', linewidth=2)
                # TODO: What if it's a different intensity?
    
        # Plot the timeseries
        sns.lineplot(data=plot_data, ax=ax[plot_id], linewidth=2)
        # TODO: Only show the x axis name ('Step') on the last plot.
        ax[plot_id].set_ylabel(col)

    # Tweak visuals
    sns.despine()
    fig.suptitle(title)

    plt.show()

################################
### PART 2: PLOTTING FOR EMA ###
################################

def plot_lines(
    experiments,
    outcomes,
    n_steps,
    outcomes_to_show=[],
    experiments_to_show=None,
    handle_floods="combine",
    group_by=None,
    highlight=None,
    show_envelope=False,
    height_per_plt=5,
    titles={},
    ):
    """Takes data from ema_workbench.perform_experiments and visualizes them
    as line plots (1 per outcome). It assumes time-series data.
    
    Parameters
    ----------
    experiments : DataFrame
                  from ema_workbench.perform_experiments
    outcomes : OutcomesDict
               from ema_workbench.perform_experiments
               assumes experiments were generated with a Model, not a ReplicatorModel
    n_steps : int
              the number of model steps (==length of the time-series data)
    outcomes_to_show : list of str, optional 
                       list of outcome names you want to plot.
                       defaults to empty list, i.e. plot all outcomes
    experiments_to_show : list of int, optional
                          list of experiment indices to include as lines
                          defaults to empty list, i.e. include all experiments
    handle_floods : str, optional
                    has two options: "combine" (default) or "separate"
                    "combine" : all experiments are taken together and floods
                                are ignored
                    "group_by" : equivalent to group_by=flood_narratives
                    "separate": experiments are separated by flood_narrative
                                and each (flood, outcome) pair is plotted on
                                its own set of axes. vertical bars are overlaid
                                to show flood times & intensities
    group_by : str, optional
               name of the column in `experiments` by which to group lines
               in a single plot. defaults to None, no grouping
    highlight : any, optional
                if used alongside group_by, will create just two groups of lines
                (one grey and one highlighted). the lines to be highlighted are
                identified by `experiments[group_by] == highlight`
                if group_by is None, this is ignored
    show_envelope : bool, optional
                    shows an envelope within the lines (if group_by is not None,
                    shows a separate envelope for each group)
    height_per_plt : int, optional
                     height per subplot in the figure (outcome)
    """

    # Include this to clean up the ReplicatorModel thing 
    # new_outcomes = {}
    # for var in outcomes:
    #     print(var)
    #     new_outcomes[var] = []
    #     for a in outcomes[var]:
    #         new_outcomes[var].append(list(a[0]))
    #     outcomes[var] = np.array(new_outcomes[var])
    
    # create TIME variable
    if 'TIME' not in outcomes:
        outcomes['TIME'] = np.array([[i for i in range(n_steps)]])

    # Treat differently based on handle_floods
    if handle_floods == "separate":
            assert('flood_narrative' in experiments.columns)
            assert(group_by != 'flood_narrative')

            flood_narratives = experiments['flood_narrative'].unique()

            fig = axes = {}
            for f, flood_string in enumerate(flood_narratives):
                # Select only experiments of interest 
                experiments_to_show = experiments[experiments['flood_narrative'] == flood_string].index
                experiments_sub = experiments.loc[experiments_to_show, :]
                outcomes_sub = {}
                for k, v in outcomes.items():
                    if k == 'TIME': 
                        continue
                    outcomes_sub[k] = v[experiments_to_show]
                # Create plot titles
                titles={}
                for outcome in outcomes_to_show:
                    titles[outcome] = f'{outcome} ({flood_string})'
                fig[f], axes[f] = plot_lines(
                    experiments_sub,
                    outcomes_sub,
                    n_steps,
                    outcomes_to_show=outcomes_to_show,
                    # experiments_to_show=experiments_to_show,
                    handle_floods="combine",
                    group_by=group_by,
                    highlight=highlight,
                    show_envelope=show_envelope,
                    height_per_plt=height_per_plt,
                    titles=titles
                )
                
                # Add flood lines TODO
                flood_dict = {
                    int(flood.split(':')[0].strip()) : int(flood.split(':')[1].strip())
                    for flood in flood_string[1:-1].split(',')
                }
                for o in outcomes_to_show:
                    # Add vertical bars to represent floods
                    for flood_time, flood_return in flood_dict.items():
                        if flood_return == 10:
                            axes[f][o].axvline(x=flood_time, color='r', linestyle=':', linewidth=1)
                        elif flood_return == 100:
                            axes[f][o].axvline(x=flood_time, color='r', linestyle='--', linewidth=1)
                        elif flood_return == 1000:
                            axes[f][o].axvline(x=flood_time, color='r', linestyle='-', linewidth=1)
                        # TODO: What if it's a different intensity?
    else: 
        if handle_floods == "combine":
            fig, axes = lines(
                experiments,
                outcomes,
                outcomes_to_show=outcomes_to_show,
                experiments_to_show=experiments_to_show,
                group_by=group_by,
                show_envelope=show_envelope,
                titles=titles,
            )
        elif handle_floods == "group_by":
            assert(group_by is None or group_by=="flood_narrative")
            fig, axes = lines(
                experiments,
                outcomes,
                outcomes_to_show=outcomes_to_show,
                experiments_to_show=experiments_to_show,
                group_by="flood_narrative",
                show_envelope=show_envelope,
                titles=titles,
            )
        else:
            # TODO: proper error handling
            print("handle_floods must be combine, group_by, or separate")
            return
    
        # Make plot more readable
        fig.set_size_inches(12, len(axes)*height_per_plt)
        for var in axes:
            axes[var].spines['right'].set_visible(False)
            axes[var].spines['top'].set_visible(False)

            for line in range(len(experiments)):
                axes[var].properties()['children'][line].set(
                    color='grey',
                    linewidth=1,
                    alpha=0.5
                )

        # Handle highlighting
        # TODO
        
    return fig, axes

def plot_name(title: str):
    
    DATE = datetime.now().strftime("%y%m%d_%H%M%S")
    return '_'.join(title.lower().split(' ')) + '__' + DATE

def save_fig(title: str, dir: str):
    # Get formatted file name
    title = plot_name(title)
    # Format path
    for format in ['png', 'svg']:
        path = f'../img/{format}/{dir}/{title}.{format}' 

        plt.savefig(path, dpi=300, format=format)