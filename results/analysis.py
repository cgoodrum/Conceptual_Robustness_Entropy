import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
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
    sns.distplot(time_data[1], bins = bins, hist = True, kde = False, label = 'Case 1 (labelled)')
    sns.distplot(time_data[2], bins = bins, hist = True, kde = False, label = 'Case 2')
    sns.distplot(time_data[3], bins = bins, hist = True, kde = False, label = 'Case 3')

    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('CPU Rework Time (s)')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Case Rework Times')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

def plot_kde(time_data, filename = None, **kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    kde_kws = {
        'shade': True
    }
    sns.distplot(time_data[1], hist = False, label = 'Case 1 (labelled)', kde_kws = kde_kws )
    sns.distplot(time_data[2], hist = False, label = 'Case 2', kde_kws = kde_kws)
    sns.distplot(time_data[3], hist = False, label = 'Case 3', kde_kws = kde_kws)

    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('CPU Rework Time (s)')
    plt.ylabel('Probability Density')
    plt.title('KDE of Case Rework Times')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
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
    plt.plot(x, time_data[2], label = 'Case 2')
    plt.plot(x, time_data[3], label = 'Case 3')
    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('Trial')
    plt.ylabel('Rework Time (s)')
    plt.title('Rework Time Case Data')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

###############################################################################
def main():

    # Define files to read in
    rework_time_files = [
    'case1_labelled_rework_time_data.csv',
    'case2_rework_time_data.csv',
    'case3_rework_time_data.csv'
    ]

    # Read files to a single dict
    case_time_data = {}
    for f in rework_time_files:
        case = int(''.join(x for x in f if x.isdigit()))
        case_time_data[case]= read_rework_time_file(f)


    # PLOT AND SAVE FILES
    plot_time_data(case_time_data, filename = 'Rework_Time_Raw_Data')
    plot_hist(case_time_data, bins = 100, filename = 'Rework_Time_Histogram' )
    plot_kde(case_time_data, filename = 'Rework_Time_Kernel_Plot' )


if __name__ == '__main__':
    main()
