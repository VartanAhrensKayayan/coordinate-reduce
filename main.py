import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random as r
import numpy as np
from math import floor
import seaborn
import PIL  # Not used but needs to be installed in order to save files as .tiff. If .png is used can be ignored.

# for color scales
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def lorenz_curver(group_size,
                  dataframe,
                  groups_reducing=1,
                  seed=None,
                  energy_counting=False,
                  maximum_reduction=100):
    """
    Instead of splitting apartments into different groups and reducing all their top loads, only a number of
    groups reduce their top load.

    :param group_size: Number of residential units that are grouped together before reducing load.
    :param dataframe: Data about electricity use for each 8760 hours (rows) in residential unit (columns).
    :param groups_reducing: Number of groups (of group size) whose top loads will be reduced.
    :param seed: int, optional. Set to a number to maintain the random shuffling the same across applications.
    :param energy_counting: bool, optional. Set to True to calculate the energy needed to cut the peaks instead of new peak.
    :param maximum_reduction: int, optional. Percentual reduction which is set as maximum (between 0 and 100).
    :return: dictionary Contains either the peak electrical power or energy reductions for each reduction percentile.
    """

    # Randomly shuffle the columns of the dataframe
    rng = np.random.default_rng(seed=seed)  # A seed can be set to reproduce results.
    dataframe = dataframe[rng.permutation(dataframe.columns.values)]

    # This generates the groups based on the group_size and the groups reducing parameters
    population_size = dataframe.shape[1]
    column_indices = np.arange(dataframe.shape[1])
    group_indices = np.array_split(column_indices, np.ceil(population_size / group_size))
    dfs = [dataframe.iloc[:, indices] for indices in group_indices]

    # Create a range of percentage reductions (0 to maximum_reduction)
    percentage = np.round(np.linspace(1, maximum_reduction - 1, maximum_reduction - 1), 4).astype(float)

    results = {}

    for percentile in percentage:
        counter = 0
        capped_data = {}  # Store capped data as a dictionary rather than pandas since it is faster to concat later.
        total_energy_reduction = 0  # To accumulate energy reduction for this percentile
        for group_idx, df in enumerate(dfs):
            if counter < groups_reducing:
                df_sum = df.sum(axis=1)
                border = df_sum.max() * ((100 - percentile) / 100)

                # Cap values at the border
                capped_df = df_sum.where(df_sum < border, border)

                # Calculate energy reduction (only if requested)
                if energy_counting:
                    total_energy_reduction = (df_sum - capped_df).sum()

                capped_data[group_idx] = capped_df
                counter += 1
            else:
                capped_data[group_idx] = df.sum(axis=1)  # Keep uncapped groups unchanged

        # Find the maximum value across all capped/uncapped groups
        capped_df_combined = pd.DataFrame(capped_data).sum(axis=1)
        max_value = capped_df_combined.max()

        # Save results if energy counting, returns energy lost else return the new peak power.
        if energy_counting:
            results[f'{percentile}'] = round(total_energy_reduction, 4)
        else:
            results[f'{percentile}'] = round(max_value, 4)
    return results


def postprocess(dictionary, save_figure_as, colors=plt.rcParams['axes.prop_cycle'].by_key()['color'] * 4):
    # This takes a dictionary with the results and makes the graph.
    # If a color map is specified it is used, otherwise it uses the standard repeated 4 times to ensure enough colors.
    # The resulting graph is then saved.
    final_lorenz = pd.DataFrame(dictionary)

    fig, ax = plt.subplots()

    i = 0
    for column in final_lorenz.columns:
        ax.plot(final_lorenz[column], label=column, c=colors[i])
        i += 1
    ax.set_xlabel('Percentile reduction in groups total load (%)')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    ax.set_ylabel('Peak Power \n (kW)', rotation=0, labelpad=30)
    ax.set_ylim(None, None)  # forces the y-axis to start at 0.
    ax.set_xlim(None, None)

    fig.set_size_inches(w=8.27, h=11.69 / 3)  # A third of an A4.
    plt.legend(title='Apartments Group Size')
    plt.tight_layout()
    plt.savefig(f'results/{save_figure_as}', dpi=1200)  # High quality for final but perhaps turn down for testing.
    print(f'done with {save_figure_as}')


def seed_effect(group_size=62, max=100):
    result_total = {}
    for seed in range(500):
        result_total[seed] = lorenz_curver(group_size=group_size,
                                           dataframe=electricity,
                                           groups_reducing=1,
                                           maximum_reduction=max)

    final_lorenz = pd.DataFrame(result_total)
    location = f'results/seed_check_{group_size}.xlsx'
    final_lorenz.to_excel(location)
    postprocess_seed_effect(location=location, save_figure_as=f'seed_check_{group_size}_max{max}.tiff')


def postprocess_seed_effect(location, save_figure_as):
    # Seed effect is postprocessed differently since it takes longer and the results are shown as boxplots.
    df = pd.read_excel(f'{location}', index_col=0)
    df = df.T
    fig, ax = plt.subplots()
    ax.boxplot(df)
    ax.set_xlabel('Percentile reduction in groups total load (%)')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    ax.set_ylabel('Peak power \n (kW)', rotation=0, labelpad=30)
    ax.set_ylim(0, None)
    fig.set_size_inches(w=8.27, h=11.69 / 3)  # A third of an A4
    plt.tight_layout()
    plt.savefig(f'results/{save_figure_as}', dpi=1200)

    print(f'done with {save_figure_as}')
    plt.close()


def same_number_different_config_279(seed=10, max=100, energy=False):
    # The seed should be fixed to ensure that the same apartments are reduced in different iterations.
    name = f'279_split_different_{max}max.tiff'
    if energy:
        name = 'energy_' + name
    result_total = {}
    result_total['1 x 279'] = lorenz_curver(group_size=279,
                                            dataframe=electricity,
                                            groups_reducing=1,
                                            seed=seed,
                                            maximum_reduction=max,
                                            energy_counting=energy)
    result_total['3 x 93'] = lorenz_curver(group_size=93,
                                           dataframe=electricity,
                                           groups_reducing=3,
                                           seed=seed,
                                           maximum_reduction=max,
                                           energy_counting=energy)
    result_total['9 x 31'] = lorenz_curver(group_size=31,
                                           dataframe=electricity,
                                           groups_reducing=9,
                                           seed=seed,
                                           maximum_reduction=max,
                                           energy_counting=energy)
    postprocess(dictionary=result_total, save_figure_as=name, colors=get_color_map([279, 93, 31]))


def same_number_different_config_186(seed=1, max=100, energy_counting=False, save_figure_as='186_split'):
    # The seed should be fixed to ensure that the same apartments are reduced in different iterations.

    save_figure_as = save_figure_as + f'_{max}max'
    if energy_counting:
        save_figure_as = 'energy_' + save_figure_as

    result_total = {}
    result_total['1 x 186'] = lorenz_curver(group_size=186, dataframe=electricity, groups_reducing=1, seed=seed,
                                            maximum_reduction=max, energy_counting=energy_counting)
    result_total['2 x 93'] = lorenz_curver(group_size=93, dataframe=electricity, groups_reducing=2, seed=seed,
                                           maximum_reduction=max, energy_counting=energy_counting)
    result_total['3 x 62'] = lorenz_curver(group_size=62, dataframe=electricity, groups_reducing=3, seed=seed,
                                           maximum_reduction=max, energy_counting=energy_counting)
    result_total['6 x 31'] = lorenz_curver(group_size=31, dataframe=electricity, groups_reducing=6, seed=seed,
                                           maximum_reduction=max, energy_counting=energy_counting)
    save_figure_as = save_figure_as + '.tiff'
    postprocess(dictionary=result_total, save_figure_as=save_figure_as, colors=get_color_map([186, 93, 62, 31]))


def all_reducing_group(max=100):
    possible_groups = [1, 2, 3, 6, 9, 18, 31, 62, 93, 186, 279, 558]
    result_total = {}
    for group in possible_groups:
        result_total[group] = lorenz_curver(group_size=group, dataframe=electricity, groups_reducing=int(558 / group),
                                            seed=10, maximum_reduction=max, energy_counting=True)
    postprocess(dictionary=result_total, save_figure_as=f'energy_all_reducing_{max}max.tiff', colors=get_color_map())


def one_group_reducing():
    possible_groups = [1, 2, 3, 6, 9, 18, 31, 62, 93, 186, 279, 558]
    result_total = {}
    for group in possible_groups:
        print(f'group running: {group}')
        result_total[group] = lorenz_curver(group_size=group,
                                            dataframe=electricity,
                                            groups_reducing=1,
                                            seed=10,
                                            maximum_reduction=100)
    postprocess(dictionary=result_total, save_figure_as='one_group_reducing.tiff')
    result = pd.DataFrame(result_total)
    result.to_excel('one_group_reducing.xlsx')


def get_color_map(values=[1, 2, 3, 6, 9, 18, 31, 62, 93, 186, 279, 558]):
    # Normalize the data
    norm = Normalize(vmin=min(values), vmax=max(values))

    # Adjust the colormap to start at an even darker shade of green, so it is visible.
    # Here green was picked but any color map can be applied.
    cmap = plt.cm.Greens
    cmap_segmented = LinearSegmentedColormap.from_list("segmented", cmap(np.linspace(0.5, 1, 256)))

    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap_segmented)

    # Generate corresponding colors for the values
    colors = [cmap_segmented(norm(val)) for val in values]
    return colors


if __name__ == "__main__":
    # Fill in where the data is available if it is in a CSV or whichever method to place it into a pandas dataframe.
    # It is important that the data is organized as columns for each individual household and rows as measurements.
    # The scale of the measurements are not important (here hourly but could be 15 minutes)
    # as long as they are simultaneous for all the household.
    electricity = pd.read_csv('YOUR_DATA_SOURCE_HERE')

    result_total = {}  # to save the results

    # Figure 2
    all_reducing_group(max=100)

    # Figure 3 takes a long time so instead of saving to the picture it is stored as xlsx then can be post-processed.
    one_group_reducing()
    results = pd.read_excel('results/one_group_reducing.xlsx', index_col=0)
    results = results.to_dict()
    postprocess(results, save_figure_as='one_group_reducing_test.tiff', colors=get_color_map())

    # Figure 4
    same_number_different_config_186(seed=9, max=100)

    # Figure 5
    same_number_different_config_186(seed=9, max=40)

    # Figure 6
    same_number_different_config_279(max=100)

    # Figure 7
    seed_effect(group_size=62)

    # Figure 8
    seed_effect(group_size=93)
