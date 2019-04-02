
import sys
sys.path.append('../')
import networkx as nx
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import math
#import pylab
import itertools
#import pygraphviz
import matplotlib.pyplot as plt
import dit
from scipy.stats import entropy
from data_sources.data_interpreter import Case_Data
import time
from collections import defaultdict
import logging
import seaborn as sns
import random as rd
import numpy as np
import pandas as pd
from copy import deepcopy

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format = '%(levelname)s: %(message)s', level=logging.INFO)
#pylab.ion()

'''
                                    TO DO
    - Look at the case of removing the second largest volume ship.
        - The time series should be the same across the cases, but the probability of selecting the correct ship should change
        significantly. Track the probability of selecting the correct ship for each of the cases.
    - Add calculate_binary_outcomes() to Case 1 and Case 3
    - Expand the calculate_binary_entropy to the whole network, rather than just individual nodes.
    - determine how to track binary entropy towards target over time
    - fix "visualize_network_growth" (not pressing right now)
    - figure out how to represent binary data_status as outcomes for dit for mutual information, etc.
    - Determine a new entropy measure combining entropy of data_status, topology, and values.
'''

class Knowledge_Network(object):
    # This class defines the knowledge network for a single agent, used to
    # demonstrate cases 1, 2, and 3.

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, target_node = 0, time_step = 0):
        self.name = name
        self.target_node = target_node
        self.time_step = time_step
        self.rework_time = float()
        self.network = self.init_network()
        self.topological_entropy_time_series = {}
        self.simple_entropy_time_series = {}
        self.binary_entropy_time_series = {}
        self.target_value_shannon_entropy_time_series = {}
        self.target_value_CRE_entropy_time_series = {}
        self.graphviz_path = 'C:/Python37/Lib/site-packages/graphviz-2.38/release/bin'

    def init_network(self):
        G = nx.DiGraph()
        G.add_node(self.target_node, **self.add_attributes('avg_vol'))
        return G

    def calculate_data_status(self, node):
        # Determines the data status of a node based on if it has a value or
        # not.
        if self.network.node[node]['val']:
            status = 1.0
        else:
            status = 0.0
        return status

    def add_attributes(self, name = '', value = None):
        if value:
            status = 1.0
        else:
            status = 0.0
        out_dict = {
            'node_name': name,
            'val': value,
            'data_status': status
        }
        return out_dict

    def calculate_pagerank(self, alpha = 0.9):
        pagerank = nx.pagerank(self.network, alpha = alpha)
        return pagerank

    def calculate_outcome_probs(self, node):
        n = len(self.calculate_binary_outcomes(node)) # Find the connected nodes to the target node
        return [1.0/float(n)]*n

    def calculate_binary_entropy(self, node):
        temp = dit.Distribution(self.calculate_binary_outcomes(node), self.calculate_outcome_probs(node))
        entropy = dit.shannon.entropy(temp)
        return entropy

    def calculate_simple_entropy(self):
        # A simple calculation using the data status of each data element, using the
        # P(1) and P(0). This ignores structure of the network.
        # Has a min value of 0, and a max value of 1

        num_nodes = len(self.network.nodes())

        if num_nodes <= 1:
            return 0

        sum_1 = 0.0
        sum_0 = 0.0
        for node in self.network.nodes(data=True):
            data_status = node[1]['data_status']
            if data_status == 1.0:
                sum_1 += 1.0
            else:
                sum_0 += 1.0

        p_1 = sum_1/float(num_nodes)
        p_0 = sum_0/float(num_nodes)
        p = [p_0, p_1]
        H = entropy(p,base=2)
        return H

    def calculate_topological_entropy(self):
        # Calculates the entropy of the network using the pagerank of each node
        # ignoring any of the data statuses or values.
        pagerank = self.calculate_pagerank()
        d = dit.ScalarDistribution(pagerank)
        entropy = dit.shannon.entropy(d)
        return entropy

    def calculate_target_value_entropy(self, method = "shannon", **kwargs):
        # Calculates the entropy of the values of the target node based on its
        # distribution

        def value_shannon_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, **kwargs)
            probs = histogram_data[0]
            H = entropy(probs, base = 2)
            return H

        def value_CRE_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, **kwargs)
            bins = histogram_data[1][1:]
            probs = histogram_data[0]*(bins[1]-bins[0])

            d = dit.ScalarDistribution(bins,probs)
            H = dit.other.cumulative_residual_entropy(d)
            return H

        node = self.target_node
        value_entropy = {
            "shannon": value_shannon_entropy(node, **kwargs),
            "CRE": value_CRE_entropy(node, **kwargs)
            # Could add more entropy measures if required!
        }
        return value_entropy[method]

    def get_histogram_data(self, node, **kwargs):
        pred = list(self.network.predecessors(node)) # Find the connected nodes
        pred = sorted([k for k in pred], reverse=True) # Sort the calculated nodes in descending order.
        values = []
        for n in pred:
            values.append(self.network.node[n]['val'])
        histogram_data = plt.hist(values, **kwargs)
        plt.close()
        return histogram_data, values

    def build_entropy_time_series(self, time_series_name, entropy_function):
        time_series_name[self.time_step] = entropy_function

    def visualize_network_growth(self):
        self.get_network()
        pylab.draw()
        plt.pause(0.2)

    def get_network(self, prog = 'twopi',**kwargs):
        prog = '{}/{}'.format(self.graphviz_path, prog)
        pos = graphviz_layout(self.network, prog = prog)
        labels = {k: self.network.nodes(data=True)[k]['node_name'] for k in self.network.nodes()}
        nx.draw(self.network, pos, labels = labels, **kwargs)

    def save_entropy_time_series_plot(self, time_series_dict, filename = 'untitled', xlabel = 'Time', ylabel = 'Entropy', title = '', x_min = 0, x_max = 15, y_min = 0, y_max=5, grid = True,**kwargs):
        plt.figure()
        sns.set()
        sns.set_style('white')
        plt.plot(time_series_dict.keys(), time_series_dict.values())
        sns.despine()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.grid(grid)
        plt.title(title)
        plt.savefig('../figures\\' + filename +'.png')

    def save_final_network_plot(self, filename = 'untitled_network', **kwargs):
        plt.figure()
        self.get_network(**kwargs)
        plt.savefig('../figures\\{}.png'.format(filename))

    def save_target_value_histogram_plot(self, filename = None, label = '', **kwargs):

        data, values = self.get_histogram_data(
            node = self.target_node,
            **kwargs['hist_kwargs']
        )

        plt.figure()
        sns.set()
        sns.set_style('white')
        sns.distplot(values, bins = kwargs['hist_kwargs']['bins'], label = label, **kwargs['plot_kwargs'])
        sns.despine()
        plt.legend(loc='upper right')
        plt.xlabel('Volume')
        plt.ylabel('Probability Density')
        plt.title('Histogram of Target Values')
        axes = plt.gca()
        axes.set_xlim([kwargs['hist_kwargs']['range'][0],kwargs['hist_kwargs']['range'][1]])
        axes.set_ylim([None,None])
        if filename:
            plt.savefig('../figures\\{}.png'.format(filename))
        else:
            plt.show()

    def grow(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class Case_1(Knowledge_Network):
    # This case computes the averages for each variable, then multiplies them
    # together to get the average volume.

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def grow(self):
        # Add the average L, B, T, and Cb nodes, as well as {X1...Xn} for each,
        # connected to target with associated edges. Import the associated data
        # for the individual variable nodes
        new_node_label = max(self.network.nodes()) + 1

        if self.time_step == 2: # Do avg_L first
            for k,v in self.data.data.items():
                self.network.add_node(new_node_label + k, **self.add_attributes('L{}'.format(k),v['L']))
                self.network.add_edge(new_node_label + k, 1)

        elif self.time_step == 3: # Then avg_B
            for k,v in self.data.data.items():
                self.network.add_node(new_node_label + k, **self.add_attributes('B{}'.format(k),v['B']))
                self.network.add_edge(new_node_label + k, 2)

        elif self.time_step == 4: # Then avg_T
            for k,v in self.data.data.items():
                self.network.add_node(new_node_label + k, **self.add_attributes('T{}'.format(k),v['T']))
                self.network.add_edge(new_node_label + k, 3)

        elif self.time_step == 5: # Then avg_Cb
            for k,v in self.data.data.items():
                self.network.add_node(new_node_label + k, **self.add_attributes('Cb{}'.format(k),v['Cb']))
                self.network.add_edge(new_node_label + k, 4)

        else:
            print("Error: Time limit exceeds number of average variable values!")

    def initial_grow(self):
        self.network.add_node(1, **self.add_attributes('avg_L'))
        self.network.add_edge(1, 0)
        self.network.add_node(2, **self.add_attributes('avg_B'))
        self.network.add_edge(2, 0)
        self.network.add_node(3, **self.add_attributes('avg_T'))
        self.network.add_edge(3, 0)
        self.network.add_node(4, **self.add_attributes('avg_Cb'))
        self.network.add_edge(4, 0)
        self.time_step = 1

    def calculate_non_target_values(self):
        # For each non target node in the network with non-zero in-degree, calculate the
        # value of the target node using the product of the values and the data statuses.
        calc_nodes = {k:v for k,v in self.network.in_degree() if (v > 0.0 and k != self.target_node)} # Find the non target nodes with non-zero in degree to be calculated
        pred = {k:self.network.predecessors(k) for k in calc_nodes.keys()} # Find the connected nodes to the calculated calc_nodes
        sorted_node_list = sorted([k for k in calc_nodes.keys()], reverse=True) # Sort the calculated nodes in descending order.
        for n in sorted_node_list:
            val = 0.0
            status_val = 1.0
            num_vars = 0
            for n1 in pred[n]:
                val += self.network.node[n1]['val']
                status_val = status_val*self.network.node[n1]['data_status']
                num_vars += 1
            val = status_val*val/float(num_vars)
            self.network.node[n]['val'] = val
            self.network.node[n]['data_status'] = self.calculate_data_status(n)

    def calculate_target_value(self):
        # Calculate the target node (average volume) given the nodes it is connected to in the
        # knowledge network using their values and data statuses
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        val = 1.0
        new_val = 1.0
        for n in pred:
            if self.network.node[n]['val'] != None:
                new_val = self.network.node[n]['val']*self.network.node[n]['data_status']
                val = val*new_val
            else:
                val = None
                break

        self.network.node[self.target_node]['val'] = val
        self.network.node[self.target_node]['data_status'] = self.calculate_data_status(self.target_node)

    def remove_bad_ship_labelled(self, print_output = False):

        if print_output:
            print()
            print("Case 1 (Labelled):")
            print()
            print("Initial val:", self.network.node[0]['val'])

        # Start the timer
        start_time = time.clock()

        # Re-organize the data to calculate a volume for each ship
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        num_vars = len(pred)
        node_data = {}
        for n in pred:
            pred_n = list(self.network.predecessors(n)) # Find the connected nodes to node n
            num_ships = len(pred_n)
            node_data[n] = {}
            for n1 in pred_n:
                #node_data[n][n1] =self.network.node[n1]['node_name'], self.network.node[n1]['val']
                if (self.network.node[n1]['node_name'][0] + self.network.node[n1]['node_name'][1]) == 'Cb':
                    node_data[n][int(self.network.node[n1]['node_name'][2:])] = n1, self.network.node[n1]['val']
                else:
                    node_data[n][int(self.network.node[n1]['node_name'][1:])] = n1, self.network.node[n1]['val']

        # Flip the data for multiplication
        node_data_flipped = defaultdict(dict)
        for n, val in node_data.items():
            for ship, subval in val.items():
                node_data_flipped[ship][n] = subval
        node_data_flipped = dict(node_data_flipped)

        # Calculate the volumes
        vol_dict = {}
        for s, d in node_data_flipped.items():
            vol_dict[s] = d[1][1]*d[2][1]*d[3][1]*d[4][1]

        # Find the largest value for removal
        bad_ship = max(vol_dict, key=vol_dict.get) # Find the node value of the bad ship (largest volume)

        # Find the nodes and edges to remove from that ship and remove them
        for n in pred:
            edges_to_remove = list(self.network.out_edges(node_data[n][bad_ship][0])) # Find out edges
            self.network.remove_edges_from(edges_to_remove) # remove bad edges from network
            self.network.remove_node(node_data[n][bad_ship][0])

        # Re-calculate the averages, then the target node
        self.calculate_non_target_values()
        self.calculate_target_value()

        # Stop timer
        end_time = time.clock()
        elapsed_time = end_time - start_time

        if print_output:
            print("Final val:", self.network.node[0]['val'])
            print("Time Elapsed:", elapsed_time)

        self.rework_time = elapsed_time

    def remove_bad_ship_unlabelled(self, print_output = False):

        if print_output:
            print("")
            print("Case 1 (Un-labelled):")
            print("")
            print("Initial val:", self.network.node[0]['val'])

        # Start the timer
        start_time = time.clock()

        # Find all possible combinations of variables
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        val_list = {}
        pred_n = {n: list(self.network.predecessors(n)) for n in pred} # Find the connected nodes to node n
        count = 0
        for n1 in pred_n[1]:
            for n2 in pred_n[2]:
                for n3 in pred_n[3]:
                    for n4 in pred_n[4]:
                        count += 1
                        val_list[(n1,n2,n3,n4)] = [self.network.node[n1]['val'],self.network.node[n2]['val'],self.network.node[n3]['val'],self.network.node[n4]['val']]

        # Compute volume for each combination
        vol_dict = {}
        for k,v in val_list.items():
            vol_dict[k] = v[0]*v[1]*v[2]*v[3]

        count = 1
        for k in sorted(vol_dict.values(), reverse = True):
            print(count, k)
            count += 1

        # Find the largest value for removal
        bad_nodes = list(max(vol_dict, key=vol_dict.get)) # Find the node value of the bad ship (largest volume)

        # Find the nodes and edges to remove from that ship and remove them
        for n in bad_nodes:
            edges_to_remove = list(self.network.out_edges(n)) # Find out edges
            self.network.remove_edges_from(edges_to_remove) # remove bad edges from network
            self.network.remove_node(n)

        # Re-calculate the averages, then the target node
        self.calculate_non_target_values()
        self.calculate_target_value()

        # Stop timer
        end_time = time.clock()
        elapsed_time = end_time - start_time

        if print_output:
            print("Final val:", self.network.node[0]['val'])
            print("Time Elapsed:", elapsed_time)

        self.rework_time = elapsed_time

    def run(self, T=5, **kwargs):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            if i == 1:
                self.initial_grow()
            else:
                self.time_step+=1
                self.grow()
            self.calculate_non_target_values()
            self.calculate_target_value()

            self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
            self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())

class Case_2(Knowledge_Network):
    # This case computes a volume for each ship, then takes the averages of the
    # volumes.

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def grow(self):
        # Add the volume node, as well as one L, B, T, and Cb node, all
        # connected to target with associated edges. Import the associated data
        # for the variable nodes
        new_node_label = max(self.network.nodes()) + 1
        self.network.add_node(new_node_label, **self.add_attributes('V'+str(self.time_step)))
        self.network.add_edge(new_node_label, 0)
        self.network.add_node(new_node_label+1, **self.add_attributes('L'+str(self.time_step),self.data.data[self.time_step]['L']))
        self.network.add_node(new_node_label+2, **self.add_attributes('B'+str(self.time_step),self.data.data[self.time_step]['B']))
        self.network.add_node(new_node_label+3, **self.add_attributes('T'+str(self.time_step),self.data.data[self.time_step]['T']))
        self.network.add_node(new_node_label+4, **self.add_attributes('Cb'+str(self.time_step),self.data.data[self.time_step]['Cb']))
        self.network.add_edge(new_node_label+1, new_node_label)
        self.network.add_edge(new_node_label+2, new_node_label)
        self.network.add_edge(new_node_label+3, new_node_label)
        self.network.add_edge(new_node_label+4, new_node_label)

    def calculate_non_target_values(self):
        # For each non target node in the network with non-zero in-degree, calculate the
        # value of the target node using the product of the values and the data statuses.
        calc_nodes = {k:v for k,v in self.network.in_degree() if (v > 0.0 and k != self.target_node)} # Find the non target nodes with non-zero in degree to be calculated
        pred = {k:self.network.predecessors(k) for k in calc_nodes.keys()} # Find the connected nodes to the calculated calc_nodes
        sorted_node_list = sorted([k for k in calc_nodes.keys()], reverse=True) # Sort the calculated nodes in descending order.
        for n in sorted_node_list:
            val = 1.0
            new_val = 1.0
            for n1 in pred[n]:
                new_val = self.network.node[n1]['val']*self.network.node[n1]['data_status']
                val = val*new_val
            self.network.node[n]['val'] = val
            self.network.node[n]['data_status'] = self.calculate_data_status(n)

    def calculate_target_value(self):
        # Calculate the target node (average volume) given the nodes it is connected to in the
        # knowledge network using their values and data statuses
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        val = 0.0
        status_val = 1.0
        for n in pred:
            val += self.network.node[n]['val']
            status_val = status_val*self.network.node[n]['data_status']
        val = status_val*val/float(len(pred))
        self.network.node[self.target_node]['val'] = val
        self.network.node[self.target_node]['data_status'] = self.calculate_data_status(self.target_node)

    def calculate_binary_outcomes(self, node):

        #For given node, enumerate all possible combinations of input data_statuses.
        pred = list(self.network.predecessors(node)) # Find the connected nodes to the target node
        n = len(pred)
        binary_inputs = ["".join(seq) for seq in itertools.product("01", repeat=n)] # create list of binary outcomes for inputs
        binary_io = []

        # Add logic for node data status based on input data statuses
        for s in binary_inputs:
            if '0' in s:
                binary_io.append("0{}".format(s))
            else:
                binary_io.append("1{}".format(s))

        return binary_io

    def remove_bad_ship(self, print_output = False):

        # if print_output:
        #     print ""
        #     print "Case 2:"
        #     print ""
        #     print "Initial val:", self.network.node[0]['val']

        # Start the timer
        start_time = time.clock()

        # Find and remove the bad nodes and edges
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        node_data = {}
        for n in pred:
            node_data[n] = self.network.node[n]['val']
        bad_ship_node = max(node_data, key=node_data.get) # Find the node value of the bad ship (largest volume)
        pred_bad_ship = list(self.network.predecessors(bad_ship_node)) # Find the connected nodes to the target node
        edges_to_remove = list(self.network.in_edges(bad_ship_node)) # Find in edges
        edges_to_remove.append(*list(self.network.out_edges(bad_ship_node))) # Add out edges
        nodes_to_remove = pred_bad_ship
        nodes_to_remove.append(bad_ship_node) # Create list of nodes to remove including bad ship node
        self.network.remove_nodes_from(nodes_to_remove) # remove bad nodes from network
        self.network.remove_edges_from(edges_to_remove) # remove bad edges from network

        # Re-calculate the target node
        self.calculate_target_value()

        # Stop timer
        end_time = time.clock()
        elapsed_time = end_time - start_time

        # if print_output:
        #     print "Final val:", self.network.node[0]['val']
        #     print "Time Elapsed:", elapsed_time

        self.rework_time = elapsed_time

    def run(self, T = 14, **kwargs):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            self.time_step+=1
            self.grow()
            self.calculate_non_target_values()
            self.calculate_target_value()

            hist_kwargs = kwargs['hist_kwargs']

            self.build_entropy_time_series(
                self.target_value_shannon_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "shannon",
                    **hist_kwargs
                )
            )

            self.build_entropy_time_series(
                self.target_value_CRE_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "CRE",
                    **hist_kwargs
                )
            )


            self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
            self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())

class Case_3(Knowledge_Network):
    # This case only imports the volumes associated with each ship, without
    # the associated variable values.

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def grow(self):
        # Add the volume node, and associated data, and connect to the target
        # node (average volume).
        new_node_label = max(self.network.nodes()) + 1
        self.network.add_node(new_node_label, **self.add_attributes('V'+str(self.time_step), value = self.data[self.time_step]['V']))
        self.network.add_edge(new_node_label, 0)

    def calculate_target_value(self):
        # Calculate the target node (average volume) given the nodes it is connected to in the
        # knowledge network using their values and data statuses
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        val = 0.0
        status_val = 1.0
        for n in pred:
            val += self.network.node[n]['val']
            status_val = status_val*self.network.node[n]['data_status']
        val = status_val*val/float(len(pred))
        self.network.node[self.target_node]['val'] = val
        self.network.node[self.target_node]['data_status'] = self.calculate_data_status(self.target_node)

    def remove_bad_ship(self, print_output = False):

        # if print_output:
        #     print ""
        #     print "Case 3:"
        #     print ""
        #     print "Initial val:", self.network.node[0]['val']

        # Start the timer
        start_time = time.clock()

        #print "Initial val:", self.network.node[0]['val']

        # Find and remove the bad nodes and edges
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        node_data = {}
        for n in pred:
            node_data[n] = self.network.node[n]['val']
        bad_ship_node = max(node_data, key=node_data.get) # Find the node value of the bad ship (largest volume)
        edges_to_remove = list(self.network.out_edges(bad_ship_node)) # Find out edges
        self.network.remove_node(bad_ship_node) # remove bad nodes from network
        self.network.remove_edges_from(edges_to_remove) # remove bad edges from network

        # Re-calculate the target node
        self.calculate_target_value()

        # Stop timer
        end_time = time.clock()
        elapsed_time = end_time - start_time

        # if print_output:
        #     print "Final val:", self.network.node[0]['val']
        #     print "Time Elapsed:", elapsed_time

        self.rework_time = elapsed_time

    def run(self, T=14, **kwargs):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            self.time_step+=1
            self.grow()
            self.calculate_target_value()

            hist_kwargs = kwargs['hist_kwargs']

            self.build_entropy_time_series(
                self.target_value_shannon_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "shannon",
                    **hist_kwargs
                )
            )

            self.build_entropy_time_series(
                self.target_value_CRE_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "CRE",
                    **hist_kwargs
                )
            )

            self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
            self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())

class Case_4(Knowledge_Network):

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        #self.network = self.init_network()
        self.data = data
        self.average_vol_data = {}

    def grow(self):
        # Add the average L, B, T, and Cb nodes, as well as {X1...Xn} for each,
        # connected to target with associated edges. Import the associated data
        # for the individual variable nodes
        new_node_label = max(self.network.nodes()) # +1
        self.network.add_node(new_node_label+1, **self.add_attributes('L'+str(self.time_step-1),self.data.data[self.time_step-1]['L']))
        self.network.add_node(new_node_label+2, **self.add_attributes('B'+str(self.time_step-1),self.data.data[self.time_step-1]['B']))
        self.network.add_node(new_node_label+3, **self.add_attributes('T'+str(self.time_step-1),self.data.data[self.time_step-1]['T']))
        self.network.add_node(new_node_label+4, **self.add_attributes('Cb'+str(self.time_step-1),self.data.data[self.time_step-1]['Cb']))
        self.network.add_edge(new_node_label+1, 1)
        self.network.add_edge(new_node_label+2, 2)
        self.network.add_edge(new_node_label+3, 3)
        self.network.add_edge(new_node_label+4, 4)

    def initial_grow(self):
        self.network.add_node(1, **self.add_attributes('avg_L'))
        self.network.add_edge(1, 0)
        self.network.add_node(2, **self.add_attributes('avg_B'))
        self.network.add_edge(2, 0)
        self.network.add_node(3, **self.add_attributes('avg_T'))
        self.network.add_edge(3, 0)
        self.network.add_node(4, **self.add_attributes('avg_Cb'))
        self.network.add_edge(4, 0)
        self.time_step = 1

    def calculate_non_target_values(self):
        # For each non target node in the network with non-zero in-degree, calculate the
        # value of the target node using the product of the values and the data statuses.
        calc_nodes = {k:v for k,v in self.network.in_degree() if (v > 0.0 and k != self.target_node)} # Find the non target nodes with non-zero in degree to be calculated
        pred = {k:self.network.predecessors(k) for k in calc_nodes.keys()} # Find the connected nodes to the calculated calc_nodes
        sorted_node_list = sorted([k for k in calc_nodes.keys()], reverse=True) # Sort the calculated nodes in descending order.
        for n in sorted_node_list:
            val = 0.0
            status_val = 1.0
            num_vars = 0
            for n1 in pred[n]:
                val += self.network.node[n1]['val']
                status_val = status_val*self.network.node[n1]['data_status']
                num_vars += 1
            val = status_val*val/float(num_vars)
            self.network.node[n]['val'] = val
            self.network.node[n]['data_status'] = self.calculate_data_status(n)

    def calculate_target_value(self):
        # Calculate the target node (average volume) given the nodes it is connected to in the
        # knowledge network using their values and data statuses
        pred = list(self.network.predecessors(self.target_node)) # Find the connected nodes to the target node
        val = 1.0
        new_val = 1.0
        for n in pred:
            if self.network.node[n]['val'] != None:
                new_val = self.network.node[n]['val']*self.network.node[n]['data_status']
                val = val*new_val
            else:
                val = None
                break

        self.network.node[self.target_node]['val'] = val
        self.network.node[self.target_node]['data_status'] = self.calculate_data_status(self.target_node)

    def calculate_target_value_entropy(self, method = "shannon", **kwargs):
        # Calculates the entropy of the values of the target node based on its
        # distribution

        def value_shannon_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, **kwargs)
            probs = histogram_data[0]
            H = entropy(probs, base = 2)
            return H

        def value_CRE_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, target_values = self.average_vol_data, **kwargs)
            bins = histogram_data[1][1:]
            probs = histogram_data[0]*(bins[1]-bins[0])

            d = dit.ScalarDistribution(bins,probs)
            H = dit.other.cumulative_residual_entropy(d)
            return H

        node = self.target_node
        value_entropy = {
            "shannon": value_shannon_entropy(node, **kwargs),
            "CRE": value_CRE_entropy(node, **kwargs)
            # Could add more entropy measures if required!
        }
        return value_entropy[method]

    def get_histogram_data(self, node, target_values = {}, **kwargs):
        pred = list(self.network.predecessors(node)) # Find the connected nodes
        pred = sorted([k for k in pred], reverse=True) # Sort the calculated nodes in descending order.
        value = 1.0
        for n in pred:
            value = self.network.node[n]['val']*value
        target_values[self.time_step] = value
        # histogram_data = plt.hist(target_values.values(), **kwargs)
        # plt.close()
        # return histogram_data, target_values.values()
        values = list(target_values.values())
        histogram_data = plt.hist(values, **kwargs)
        plt.close()
        return histogram_data, values

    def run(self, T=15, **kwargs):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            if i == 1:
                self.initial_grow()
            else:
                self.time_step+=1
                self.grow()
                self.calculate_non_target_values()
                self.calculate_target_value()

                hist_kwargs = kwargs['hist_kwargs']

                self.build_entropy_time_series(
                    self.target_value_shannon_entropy_time_series,
                    self.calculate_target_value_entropy(
                        method = "shannon",
                        **hist_kwargs
                    )
                )

                self.build_entropy_time_series(
                    self.target_value_CRE_entropy_time_series,
                    self.calculate_target_value_entropy(
                        method = "CRE",
                        **hist_kwargs
                    )
                )

            self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
            self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())

############################# GLOBAL FUNCTIONS ################################
def save_entropy_comparison_plots(case_time_series_dict, filename = 'untitled', xlabel = 'Time', ylabel = 'Entropy', title = '', x_min = 0, x_max = 15, y_min = -0.1, y_max=5, grid = True,**kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    for k,v in case_time_series_dict.items():
        plt.plot(v.keys(), v.values(), label = 'Case {}'.format(k))
    sns.despine()
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.grid(grid)
    plt.title(title)
    plt.savefig('../figures\\' + filename +'.png')

def save_histogram_comparison_plots(case_histograms, filename = 'untitled', xlabel = 'Volume', ylabel = 'PDF', title = '', x_min = 0, x_max = 15, y_min = -0.1, y_max=5, grid = True,**kwargs):
    plt.figure()
    sns.set()
    sns.set_style('white')
    for k, v in case_histograms.items():
        sns.distplot(v[1], bins = kwargs['hist_kwargs']['bins'], label = 'Case {}'.format(k), **kwargs['plot_kwargs'])
    sns.despine()
    plt.legend(loc='upper right')
    plt.xlabel('Volume')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Target Values')
    axes = plt.gca()
    axes.set_xlim([kwargs['hist_kwargs']['range'][0],kwargs['hist_kwargs']['range'][1]])
    axes.set_ylim([None,None])
    plt.savefig('../figures\\' + filename +'.png')

def case_run_manager(cases = [], data = None, T = None, save_entropy_time_series = False, save_histograms = False, save_network = False, do_rework = False, save_comparisons = False, **kwargs):

    case = {}
    for case_num in cases:
        logging.info('_________ CASE {} _________'.format(case_num))
        logging.info(' Attempting to load Case {}'.format(case_num))

        if case_num == 1:
            case[case_num] = Case_1(data = data)
        elif case_num == 2:
            case[case_num] = Case_2(data = data)
        elif case_num == 3:
            case[case_num] = Case_3(data = data.calc_vols())
        elif case_num == 4:
            case[case_num] = Case_4(data = data)
        else:
            logging.ERROR(' Invalid Case Number')
            return

        logging.info(' Loaded Case {}'.format(case_num))
        logging.info(' Starting Case {} run with T={}'.format(case_num, T))

        if T == None:
            case[case_num].run(**kwargs)
        else:
            case[case_num].run(T=T, **kwargs)

        logging.info(' Case {} Run Complete'.format(case_num))

        if do_rework:
            logging.info(' Starting Case {} Rework'.format(case_num))
            if case_num == 1:
                if labelled:
                    case[case_num].remove_bad_ship_labelled()
                else:
                    case[case_num].remove_bad_ship_unlabelled()
            else:
                case[case_num].remove_bad_ship()
            logging.info(' Case {} rework complete'.format(case_num))

        if save_entropy_time_series:
            logging.info(' Saving Time Series Images')
            # Plot time series and save file in figures directory
            case[case_num].save_entropy_time_series_plot(
                case[case_num].topological_entropy_time_series,
                filename ='case{}_topological_entropy_time_series'.format(case_num),
                title = 'case{}_topological_entropy_time_series'.format(case_num),
                x_min = 0,
                x_max = 5,
                y_min = -0.1,
                y_max = 5,
                grid = False
            )

            case[case_num].save_entropy_time_series_plot(
                case[case_num].simple_entropy_time_series,
                filename ='case{}_simple_entropy_time_series'.format(case_num),
                title = 'case{}_simple_entropy_time_series'.format(case_num),
                x_min = 0,
                x_max = 5,
                y_min = -0.1,
                y_max = 5,
                grid = False
            )

            case[case_num].save_entropy_time_series_plot(
                case[case_num].target_value_shannon_entropy_time_series,
                filename ='case{}_target_value_entropy_time_series_shannon'.format(case_num),
                title = 'Case {} Target Value Entropy Time Series (Shannon)'.format(case_num),
                x_min = 0,
                x_max = 15,
                y_min = -0.1,
                y_max = None,
                grid = False
            )

            case[case_num].save_entropy_time_series_plot(
                case[case_num].target_value_CRE_entropy_time_series,
                filename ='case{}_target_value_entropy_time_series_CRE'.format(case_num),
                title = 'Case {} Target Value Entropy Time Series (CRE)'.format(case_num),
                x_min = 0,
                x_max = 15,
                y_min = -0.1,
                y_max = None,
                grid = False
            )

        if save_network:
            logging.info(' Saving Final Network Image')
            # Plot final network layout
            case[case_num].save_final_network_plot(
                filename = "case{}_final_network".format(case_num),
                prog = 'twopi',
                with_labels = True,
                font_weight = 'normal',
                font_size = 10,
                node_size = 120,
                node_color = 'blue',
                alpha = 0.6,
                arrowstyle = '->',
                arrowsize = 10
            )

        if save_histograms:
            logging.info(' Saving Histograms')

            if case_num == 1:
                pass # CANT CALCULATE A DISTRIBUTION FOR CASE 1 (one point)

            elif case_num == 4:
                case[case_num].save_target_value_histogram_plot(
                    filename = 'case{}_target_value_distribution'.format(case_num),
                    label = 'Case {}'.format(case_num),
                    **kwargs
                )
            else:
                case[case_num].save_target_value_histogram_plot(
                    filename = 'case{}_target_value_distribution'.format(case_num),
                    label = 'Case {}'.format(case_num),
                    **kwargs
                )

    logging.info('_________________________')

    if save_comparisons:
        # Save Case Comparison Graphs
        logging.info(' Starting Entropy Comparison Plots')

        topological_entropy_time_series_dict = {
            k:v.topological_entropy_time_series for k,v in case.items()
        }

        simple_entropy_time_series_dict = {
            k:v.simple_entropy_time_series for k,v in case.items()
        }

        target_value_shannon_entropy_time_series_dict = {
            k:v.target_value_shannon_entropy_time_series for k,v in case.items()
        }

        target_value_CRE_entropy_time_series_dict = {
            k:v.target_value_CRE_entropy_time_series for k,v in case.items()
        }

        histogram_data_dict = {
            k:v.get_histogram_data(node = v.target_node, **kwargs['hist_kwargs'])
            for k,v in case.items() if k != 1
        }

        save_entropy_comparison_plots(
            topological_entropy_time_series_dict,
            filename ='topological_entropy_time_series_comparison',
            title = 'Topological Entropy Case Comparison',
            x_min = 0,
            x_max = 15,
            y_min = -0.1,
            y_max = 5,
            grid = False
        )

        save_entropy_comparison_plots(
            simple_entropy_time_series_dict,
            filename ='simple_entropy_time_series_comparison',
            title = 'Simple Entropy Case Comparison',
            x_min = 0,
            x_max = 15,
            y_min = -0.1,
            y_max = 5,
            grid = False
        )

        save_entropy_comparison_plots(
            target_value_shannon_entropy_time_series_dict,
            filename ='target_value_entropy_time_series_comparison_shannon',
            title = 'Target Value Entropy Case Comparison (Shannon)',
            x_min = 0,
            x_max = 15,
            y_min = -0.1,
            y_max = None,
            grid = False
        )

        save_entropy_comparison_plots(
            target_value_CRE_entropy_time_series_dict,
            filename ='target_value_entropy_time_series_comparison_CRE',
            title = 'Target Value Entropy Case Comparison (CRE)',
            x_min = 0,
            x_max = 15,
            y_min = -0.1,
            y_max = None,
            grid = False
        )

        logging.info(' Starting Histogram Comparison Plot')

        save_histogram_comparison_plots(
            histogram_data_dict,
            filename = "target_value_distribution_comparison",
            title = "Target Value Distribution Case Comparison",
            **kwargs
        )

    logging.info(' Done.')
    return case

def simulate_rework_times(cases = [], data = None, num_trials = 500, save_files = False, **kwargs):
    file_path = "../results\\"
    case = {}

    for case_num in cases:

        logging.info(' Attempting to load Case {}'.format(case_num))
        if case_num == 1:
            case[case_num] = Case_1(data = data)
        elif case_num == 2:
            case[case_num] = Case_2(data = data)
        elif case_num == 3:
            case[case_num] = Case_3(data = data.calc_vols())
        # elif case_num == 4:
        #     case[case_num] = Case_4(data = data)
        else:
            logging.ERROR(' Invalid Case Number')
            return

        logging.info(' Loaded Case {}'.format(case_num))

        case_rework_times = {}

        if case_num == 1:
            # Labelled Case
            filename = "{}case1_labelled_rework_time_data_test.csv".format(file_path)
            logging.info('Starting Case 1 (labelled)')
            for i in range(num_trials):
                c = case[case_num]
                c.run(**kwargs)
                c.remove_bad_ship_labelled()
                case_rework_times[i] = c.rework_time
                logging.info(' Iteration {}, Value: {}'.format(i, c.rework_time))


            if save_files:
                logging.info('Saving file...')
                with open(filename, 'w') as f:
                    for key in case_rework_times.keys():
                        f.write("{},{}\n".format(key, case_rework_times[key]))

            logging.info('Finished Case 1 (labelled)')

            case_rework_times = {}
            filename = "{}case1_unlabelled_rework_time_data_test.csv".format(file_path)
            logging.info('Starting Case 1 (unlabelled)')
            for i in range(num_trials):
                c = case[case_num]
                #print c
                c.run(**kwargs)
                c.remove_bad_ship_unlabelled()
                case_rework_times[i] = c.rework_time
                logging.info(' Iteration {}, Value: {}'.format(i, c.rework_time))

            if save_files:
                logging.info('Saving file...')
                with open(filename, 'w') as f:
                    for key in case_rework_times.keys():
                        f.write("{},{}\n".format(key, case_rework_times[key]))

            logging.info('Finished Case 1 (unlabelled)')

        else:
            case_rework_times = {}
            filename = "{}case{}_rework_time_data_test.csv".format(file_path, case_num)
            logging.info('Starting Case {}'.format(case_num))
            for i in range(num_trials):
                c = case[case_num]
                c.run(**kwargs)
                c.remove_bad_ship_unlabelled()
                case_rework_times[i] = c.rework_time
                logging.info(' Iteration {}, Value: {}'.format(i, c.rework_time))

            if save_files:
                logging.info('Saving file...')
                with open(filename, 'w') as f:
                    for key in case_rework_times.keys():
                        f.write("{},{}\n".format(key, case_rework_times[key]))

            logging.info('Finished Case {}'.format(case_num))

def save_pickle(dataframe, filename):
    dataframe.to_pickle('../results\\{}.pkl'.format(filename))

def get_pickle(filename):
    return pd.read_pickle('../results\\{}.pkl'.format(filename))

def shift_time(row):
    if row['Case'] == 'Case 4':
        val = row['Time'] - 1
    else:
        val = row['Time']
    return val

###############################################################################
def main():

    # Import the data
    raw_data = Case_Data('../data_sources\\case_data.csv')

    ##################### SINGLE INSTANCE ###########################
    # case = case_run_manager(
    #     cases = [1,2,3,4],
    #     data = raw_data,
    #     #T = 5,
    #     do_rework = False,
    #     save_entropy_time_series = True,
    #     save_network = True,
    #     save_histograms = False,
    #     save_comparisons = True,
    #     **{
    #         'hist_kwargs': {
    #             'bins': np.linspace(1000,15000,51), # min, max, n_bins. MUST MATCH RANGE!
    #             'range': (1000,15000),
    #             'density': True,
    #             'cumulative': False
    #         },
    #
    #         'plot_kwargs': {
    #             'hist': True,
    #             'kde': True,
    #             'norm_hist': True
    #         }
    #     }
    # )



    # fmri = sns.load_dataset("fmri")
    #
    # print(fmri)
    # print(fmri.dtypes)
    #
    # plt.figure()
    # ax = sns.lineplot(
    #     x="timepoint",
    #     y="signal",
    #     hue="event",
    #     units="subject",
    #     estimator=None,
    #     lw=1,
    #     data=fmri
    # )
    # plt.show()


    ############### RANDOMIZED TRIALS ##################

    num_trials = 250
    random_CRE_target_value_entropy_time_series = {}
    random_shannon_target_value_entropy_time_series = {}
    random_target_value_entropy_time_series = {}
    data_dict = {}
    for i in range(num_trials):
        logging.info('-----------------------------------------------')
        logging.info('              Iteration {} of {}'.format(i+1, num_trials))
        logging.info('-----------------------------------------------')

        raw_data.randomize()

        case = case_run_manager(
            cases = [2,4],
            data = raw_data,
            #T = 5,
            do_rework = False,
            save_entropy_time_series = False,
            save_network = False,
            save_histograms = False,
            save_comparisons = False,
            **{
                'hist_kwargs': {
                    'bins': np.linspace(1000,15000,51), # min, max, n_bins. MUST MATCH RANGE!
                    'range': (1000,15000),
                    'density': True,
                    'cumulative': False
                },

                'plot_kwargs': {
                    'hist': True,
                    'kde': True,
                    'norm_hist': True
                }
            }
        )

        # random_CRE_target_value_entropy_time_series[i] = {
        #     k:v.target_value_CRE_entropy_time_series for k,v in case.items()
        # }
        #
        # #print(random_CRE_target_value_entropy_time_series)
        #
        # random_shannon_target_value_entropy_time_series[i] = {
        #     k:v.target_value_shannon_entropy_time_series for k,v in case.items()
        # }

        random_target_value_entropy_time_series[i] = {
            k:(deepcopy(v.target_value_CRE_entropy_time_series), deepcopy(v.target_value_shannon_entropy_time_series))
            for k,v in case.items()
        }

        data_dict[i] = deepcopy(raw_data.data)

    # Create pandas dataframe to hold the above dicts, for plotting with the errorbands

    dfs = []
    for trial, case_dict in random_target_value_entropy_time_series.items():
        tdfs = []
        raw_data = data_dict[trial]
        for case, time_dict in case_dict.items():
            data = {'Time': [], 'CRE': [], 'Shannon': [], 'Ship': [], 'L': [], 'B':[], 'T':[], 'Cb':[], 'Vol': []}
            for t, v in time_dict[0].items():
                data['Time'].append(t)
                data['CRE'].append(v)
            for t, v in time_dict[1].items():
                data['Shannon'].append(v)
            for _, v in sorted([(k, v) for k, v in raw_data.items()], key=lambda x: x[0]):
                for k1, v1 in v.items():
                    data[k1].append(v1)
                data['Vol'].append(v['L']*v['B']*v['T']*v['Cb'])
            df = pd.DataFrame(data)
            df['Vol Rank'] = df['Vol'].rank(ascending = False)
            df['Vol Rank'] = df['Vol Rank'].astype('int64')
            df['Case'] = "Case {}".format(case)
            tdfs.append(df)
        tdf = pd.concat(tdfs, ignore_index=True, sort=False)
        tdf['Trial'] = trial
        dfs.append(tdf)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    save_pickle(df, 'cases24_ensemble_data_n=250')


    # df = get_pickle('cases24_ensemble_data_n=100')
    #
    # df['Shifted Time'] = df.apply(shift_time, axis=1)
    #
    #
    # for i in range(1,15):
    #
    #     trials = df[(df['Shifted Time'] == i) & (df['Vol Rank'] == 1) & (df['Case'] == 'Case 2')]['Trial']
    #
    #     plt_df = pd.concat(
    #     [df[df['Trial'] == t] for t in trials],
    #     ignore_index = True, sort = False
    #     )
    #
    #     plt.figure()
    #     sns.set()
    #     sns.set_style('white')
    #     sns.lineplot(
    #         x = 'Shifted Time',
    #         y = 'CRE',
    #         hue = 'Case',
    #         ci = 100,
    #         n_boot = 10000,
    #         data = plt_df
    #     )
    #     sns.lineplot(
    #         x = 'Shifted Time',
    #         y = 'CRE',
    #         hue = 'Case',
    #         units = 'Trial',
    #         estimator = None,
    #         data = plt_df,
    #         **{
    #             'linewidth' : 0,
    #             'marker': '.'
    #         }
    #     )
    #     sns.despine()
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     plt.gca().legend(handles=handles[1:3], labels=labels[1:3])
    #     plt.xlabel('Time Steps from initial')
    #     plt.ylabel('Cumulative Residual Entropy (CRE)')
    #     plt.title('CRE Time Series Enseble - Outlier (t = {})'.format(i))
    #     axes = plt.gca()
    #     axes.set_xlim([0,None])
    #     axes.set_ylim([None,6500])
    #     plt.savefig('../figures\\' + 'CRE_Ensemble_Time_Series_t={}.png'.format(i))
    #     #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))







    # simulate_rework_times(
    #     cases = [1,2,3],
    #     data = raw_data,
    #     num_trials = 10,
    #     save_files = True
    # )

    # # CASE 1 (labelled)
    # filename = "../results\\case1_labelled_rework_time_data.csv"
    # case1_labelled_rework_times = {}
    #
    # for i in range(num_trials):
    #     case1 = Case_1(data = raw_data)
    #     case1.run()
    #     case1.remove_bad_ship_labelled()
    #     case1_labelled_rework_times[i] = case1.rework_time
    #     logging.info(' Iteration {}, Value: {}'.format(i, case1.rework_time))
    #
    # with open(filename, 'w') as f:
    #     for key in case1_labelled_rework_times.keys():
    #         f.write("{},{}\n".format(key, case1_labelled_rework_times[key]))
    #
    # #CASE 1 (unlabelled)
    # filename = "../results\\case1_unlabelled_rework_time_data.csv"
    # case1_unlabelled_rework_times = {}
    #
    # for i in range(num_trials):
    #     case1 = Case_1(data = raw_data)
    #     case1.run()
    #     case1.remove_bad_ship_unlabelled()
    #     case1_unlabelled_rework_times[i] = case1.rework_time
    #     logging.info(' Iteration {}, Value: {}'.format(i, case1.rework_time))
    #
    # with open(filename, 'w') as f:
    #     for key in case1_unlabelled_rework_times.keys():
    #         f.write("{},{}\n".format(key, case1_unlabelled_rework_times[key]))
    #
    # # CASE 2
    # filename = "../results\\case2_rework_time_data.csv"
    # case2_rework_times = {}
    #
    # for i in range(num_trials):
    #     case2 = Case_2(data = raw_data)
    #     case2.run(T=14)
    #     case2.remove_bad_ship()
    #     case2_rework_times[i] = case2.rework_time
    #     logging.info(' Iteration {}, Value: {}'.format(i, case2.rework_time))
    #
    # with open(filename, 'w') as f:
    #     for key in case2_rework_times.keys():
    #         f.write("{},{}\n".format(key, case2_rework_times[key]))
    #
    # # CASE 3
    # filename = "../results\\case3_rework_time_data.csv"
    # case3_rework_times = {}
    # case3_data = raw_data.calc_vols()
    #
    # for i in range(num_trials):
    #     case3 = Case_3(data = case3_data)
    #     case3.run()
    #     case3.remove_bad_ship()
    #     case3_rework_times[i] = case3.rework_time
    #     logging.info(' Iteration {}, Value: {}'.format(i, case3.rework_time))
    #
    # with open(filename, 'w') as f:
    #     for key in case3_rework_times.keys():
    #         f.write("{},{}\n".format(key, case3_rework_times[key]))

if __name__ == '__main__':
    main()
