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
    sns.distplot(time_data[1], bins = bins, hist = True, kde = False, label = 'Case 1 (labelled)')
    sns.distplot(time_data[2], bins = bins, hist = True, kde = False, label = 'Case 1 (un-labelled)')
    sns.distplot(time_data[3], bins = bins, hist = True, kde = False, label = 'Case 2')
    sns.distplot(time_data[4], bins = bins, hist = True, kde = False, label = 'Case 3')
    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('CPU Rework Time (s)')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Case Rework Times')
    axes = plt.gca()
    axes.set_xlim([0,None])
    axes.set_ylim([None,None])
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
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
    sns.distplot(time_data[2], hist = False, label = 'Case 1 (un-labelled)', kde_kws = kde_kws )
    sns.distplot(time_data[3], hist = False, label = 'Case 2', kde_kws = kde_kws)
    sns.distplot(time_data[4], hist = False, label = 'Case 3', kde_kws = kde_kws)
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

    # test = {
    #     'data': data_vals,
    #     'Case': col_vals
    # }

    test = pd.DataFrame(data_vals, columns = ['data'])
    test2 = pd.DataFrame(col_vals, columns = ['case'])
    test = test.join(test2)
    #print test


    # plot_data = pd.DataFrame(time_data[1], columns=['data'])
    # case1 = case1.append(pd.DataFrame(time_data[2], columns=['data']), ignore_index = True)
    # print case1
    # case3 = pd.DataFrame(time_data[3], columns=['data'])
    # case4 = pd.DataFrame(time_data[4], columns=['data'])
    # for c in time_data.keys():
    #
    # print test

    # plot_data['case'] = [columns[1]]*len(time_data[1])
    # plot_data['data'] = time_data[1].append(time_data[2].append(time_data[3].append(time_data[4])))
    #
    # test = pd.DataFrame(plot_data)
    # print test
    # new_time_data = {}
    # columns = {
    #     1: 'Case 1 (labelled)',
    #     2: 'Case 1 (unlabelled)',
    #     3: 'Case 2',
    #     4: 'Case 3'
    # }
    #
    #
    # case_data = {}
    # input_data = []
    # for k, v in time_data.items():
    #     input_data.append(v)
    #     new_time_data['data'] = v
    #     case_data['Case'] = [columns[k]]*len(v)
    #
    #
    # plot_data = pd.DataFrame.from_dict(new_time_data)
    # plot_case_data = pd.DataFrame.from_dict(case_data)
    #
    # print plot_case_data
    #
    # plot_data.join(plot_case_data)
    # print plot_data

    plt.figure()
    sns.set()
    sns.set_style('white')
    axes = plt.gca()
    axes.set_yscale("log")
    axes.yaxis.grid(True)
    axes.set(ylabel = '')
    #plt.ylabel('')
    #plt.xlabel('')
    plt.title('Boxplots of Rework Time')
    sns.despine()
    sns.boxplot(x = 'case', y = 'data', data=test, linewidth = 0.5)

    if filename:
        plt.savefig('../figures\\{}.png'.format(filename))
    else:
        plt.show()

###############################################################################
def main():


    # Define files to read in
    rework_time_files = {
        1: 'case1_labelled_rework_time_data.csv',
        2: 'case1_unlabelled_rework_time_data.csv',
        3: 'case2_rework_time_data.csv',
        4: 'case3_rework_time_data.csv'
    }

    # Read files to a single dict
    case_time_data = {}
    for k,v in rework_time_files.items():
        #case = int(''.join(x for x in f if x.isdigit()))
        case_time_data[k]= read_rework_time_file(v)


    # PLOT AND SAVE FILES
    plot_time_data(case_time_data, filename = 'Rework_Time_Raw_Data')
    #plot_hist(case_time_data, bins = 100, filename = 'Rework_Time_Histogram_test' )
    #plot_kde(case_time_data, filename = 'Rework_Time_Kernel_Plot_test' )
    plot_boxplots(case_time_data, filename = 'Rework_Time_Boxplots')

if __name__ == '__main__':
    main()
