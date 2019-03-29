import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
################################ FUNCTIONS ####################################

def read_rework_time_file(filename):
    data = []
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(float(row[1]))
        return data

def calc_mean(list):
    return np.mean(list)

def calc_std(list):
    return np.std(list)

def plot_hist(time_data, filename = None, bins = 10, **kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    sns.distplot(time_data[1], bins = bins, label = 'Case 1 (labelled)', **kwargs)
    sns.distplot(time_data[2], bins = bins, label = 'Case 1 (un-labelled)', **kwargs)
    sns.distplot(time_data[3], bins = bins, label = 'Case 2', **kwargs)
    sns.distplot(time_data[4], bins = bins, label = 'Case 3', **kwargs)
    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('CPU Rework Time (s)')
    plt.ylabel('Count')
    plt.title('Histogram of Case Rework Times')
    plt.xscale('log')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

def plot_kde(time_data, filename = None, bins = 10, **kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    # kde_kws = {
    #     'shade': True
    # }
    sns.distplot(time_data[1], bins = bins, label = 'Case 1 (labelled)', **kwargs)
    sns.distplot(time_data[2], bins = bins, label = 'Case 1 (un-labelled)', **kwargs)
    sns.distplot(time_data[3], bins = bins, label = 'Case 2', **kwargs)
    sns.distplot(time_data[4], bins = bins, label = 'Case 3', **kwargs)
    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('CPU Rework Time (s)')
    plt.ylabel('Probability Density')
    plt.title('KDE of Case Rework Times')
    plt.xscale('log')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

def plot_time_data(time_data, filename = None, **kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    x = [i for i in range(1,len(time_data[1])+1)]
    plt.plot(x, time_data[1], label = 'Case 1 (labelled)')
    plt.plot(x, time_data[2], label = 'Case 1 (unlabelled)')
    plt.plot(x, time_data[3], label = 'Case 2')
    plt.plot(x, time_data[4], label = 'Case 3')
    sns.despine()
    plt.legend(loc='center right')
    plt.xlabel('Trial')
    plt.ylabel('Rework Time (s)')
    plt.yscale("log")
    plt.title('Rework Time Case Data')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

def plot_boxplots(time_data, filename = None, **kwargs):

    # Set up pandas dataframe
    cols = {
        1: 'Case 1 (labelled)',
        2: 'Case 1 (unlabelled)',
        3: 'Case 2',
        4: 'Case 3'
    }

    col_vals = []
    data_vals = []
    for c in time_data.keys():
        data_vals += time_data[c]
        col_vals += [cols[c]]*len(time_data[c])

    test = pd.DataFrame(data_vals, columns = ['data'])
    test2 = pd.DataFrame(col_vals, columns = ['case'])
    test = test.join(test2)

    # Set up plots
    fig, ax = plt.subplots(1,1)
    sns.set()
    sns.set_style('white')
    ax.set_yscale("log")
    ax.yaxis.grid(True)
    plt.title('Box Plots of Rework Time')
    sns.despine()
    sns.boxplot(x = 'case', y = 'data', data=test, linewidth = 0.5)
    #sns.violinplot(x='case',y='data',data=test, linewidth = 0.5, scale = 'width')
    ax.set_xlabel('')
    ax.set_ylabel('Rework Time (s)')

    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

def plot_violinplots(time_data, filename = None, **kwargs):

    # Set up pandas dataframe
    cols = {
        1: 'Case 1 (labelled)',
        2: 'Case 1 (unlabelled)',
        3: 'Case 2',
        4: 'Case 3'
    }

    col_vals = []
    data_vals = []
    for c in time_data.keys():
        data_vals += time_data[c]
        col_vals += [cols[c]]*len(time_data[c])

    test = pd.DataFrame(data_vals, columns = ['data'])
    test2 = pd.DataFrame(col_vals, columns = ['case'])
    test = test.join(test2)

    # Set up figure
    fig, ax = plt.subplots(1,1)
    sns.set()
    sns.set_style('white')
    ax.set_yscale("log")
    ax.yaxis.grid(True)
    plt.title('Violin Plots of Rework Time')
    sns.despine()
    sns.violinplot(x='case',y='data',data=test, linewidth = 0.5, scale = 'width')
    ax.set_xlabel('')
    ax.set_ylabel('Rework Time (s)')

    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

###############################################################################
def main():

    save_files = True

    # Define files to read in
    rework_time_files = {
        1: 'case1_labelled_rework_time_data.csv',
        2: 'case1_unlabelled_rework_time_data.csv',
        3: 'case2_rework_time_data.csv',
        4: 'case3_rework_time_data.csv'
    }

    # Read files to a single dict
    case_time_data = {}
    range_max_vals = []
    for k,v in rework_time_files.items():
        #case = int(''.join(x for x in f if x.isdigit()))
        case_time_data[k]= read_rework_time_file(v)
        range_max_vals.append(max(case_time_data[k]))

    range_max = max(range_max_vals)
    # PLOT AND SAVE FILES
    if save_files:

        # Plot raw data
        plot_time_data(case_time_data, filename = 'Rework_Time_Raw_Data')

        # Plot histograms
        plot_hist(
            case_time_data,
            bins = np.geomspace(0.00001,range_max,101),
            filename = 'Rework_Time_Histogram',
            **{
                'hist': True,
                'kde': False,
                'norm_hist': False
            }
        )

        # Plot KDE plots
        plot_kde(
            case_time_data,
            bins = np.geomspace(0.00001,range_max,101),
            filename = 'Rework_Time_Kernel_Plot' ,
            **{
                'hist': False,
                'kde': True,
                'norm_hist': False,
                'kde_kws' : {
                    'shade': True
                }
            }
        )

        plot_boxplots(case_time_data, filename = 'Rework_Time_Box_Plots')

        plot_violinplots(case_time_data, filename = 'Rework_Time_Violin_Plots')


if __name__ == '__main__':
    main()
