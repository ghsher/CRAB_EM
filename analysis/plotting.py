import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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