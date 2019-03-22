
import sys
sys.path.append('../')
import networkx as nx
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import math
import pylab
import itertools
#import pygraphviz
import matplotlib.pyplot as plt
import dit
from scipy.stats import entropy
from data_sources.data_interpreter import Case_Data
pylab.ion()

'''
                                    TO DO
    - Conduct back propagation for rework
    - Expand the calculate_binary_entropy to the whole network, rather than just individual nodes.
    - determine how to track binary entropy towards target over time
    - Do Case 4
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
        self.network = self.init_network()
        self.topological_entropy_time_series = {}
        self.simple_entropy_time_series = {}
        self.binary_entropy_time_series = {}
        self.target_value_entropy_time_series = {}
        self.graphviz_path = 'C:/Users/cgoodrum/Anaconda2/Lib/site-packages/graphviz-2.38/release/bin'

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

        #print dit.other.cumulative_residual_entropy(temp)

        #print dit.multivariate.gk_common_information(temp)
        #print temp
        #print dit.profiles.ExtropyPartition(temp)
        #print dit.other.perplexity(temp)
        #print dit.profiles.ShannonPartition(temp)
        #print dit.profiles.DependencyDecomposition(temp)
        # plt.figure()
        # dit.profiles.SchneidmanProfile(temp).draw()
        # plt.savefig('../figures\\' + 'case2_Schneidman_Profile_test' +'.png')

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
            pred = list(self.network.predecessors(node)) # Find the connected nodes
            pred = sorted([k for k in pred], reverse=True) # Sort the calculated nodes in descending order.
            values = []
            for n in pred:
                values.append(self.network.node[n]['val'])
            #plt.figure()
            histogram_data = plt.hist(values, **kwargs)
            #plt.show()
            #plt.pause(1)
            probs = histogram_data[0]
            H = entropy(probs, base = 2)
            return H

        def value_CRE_entropy(node, **kwargs):
            pred = list(self.network.predecessors(node)) # Find the connected nodes
            pred = sorted([k for k in pred], reverse=True) # Sort the calculated nodes in descending order.
            values = []
            for n in pred:
                values.append(self.network.node[n]['val'])
            #plt.figure()
            histogram_data = plt.hist(values, **kwargs)
            #plt.show()
            #plt.pause(1)
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
        plt.plot(time_series_dict.keys(), time_series_dict.values(), '-*b')
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

    def grow(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class Case_1(Knowledge_Network):
    # This case computes the averages for each varaible, then multiplies them
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
            print "Error: Time limit exceeds number of average variable values!"

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

    def run(self, T=5):
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

    def run(self, T = 14):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            self.time_step+=1
            self.grow()
            self.calculate_non_target_values()
            self.calculate_target_value()

            self.build_entropy_time_series(
                self.target_value_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "CRE",
                    bins= 10,
                    range= (1000,15000),
                    normed = True,
                    cumulative = False
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

    def run(self, T=14):
        # Runs the case forwards (building the knowledge network)
        self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
        self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())
        for i in range(1,T+1):
            self.time_step+=1
            self.grow()
            self.calculate_target_value()
            self.build_entropy_time_series(
                self.target_value_entropy_time_series,
                self.calculate_target_value_entropy(
                    method = "CRE",
                    bins= 10,
                    range= (1000,15000),
                    normed = True,
                    cumulative = False
                )
            )
            self.build_entropy_time_series(self.topological_entropy_time_series, self.calculate_topological_entropy())
            self.build_entropy_time_series(self.simple_entropy_time_series, self.calculate_simple_entropy())


###############################################################################
def main():

    save_images = True # should the images be saved?

    # Import the data
    raw_data = Case_Data('../data_sources\\case_data.csv')


    # ##################### CASE 1 #######################
    # # Initialize Case 2
    # case1 = Case_1(data = raw_data)
    #
    # # Run Case 2 forwards
    # case1.run(T = 5)
    #
    # if save_images:
    #     # Plot time series and save file in figures directory
    #     case1.save_entropy_time_series_plot(
    #         case1.topological_entropy_time_series,
    #         filename ='case1_topological_entropy_time_series',
    #         title = 'case1_topological_entropy_time_series',
    #         x_min = 0,
    #         x_max = 5,
    #         y_min = 0,
    #         y_max = 5,
    #         grid = True
    #     )
    #
    #     case1.save_entropy_time_series_plot(
    #         case1.simple_entropy_time_series,
    #         filename ='case1_simple_entropy_time_series',
    #         title = 'case1_simple_entropy_time_series',
    #         x_min = 0,
    #         x_max = 5,
    #         y_min = 0,
    #         y_max = 5,
    #         grid = True
    #     )
    #
    #     # Plot final network layout
    #     case1.save_final_network_plot(
    #         filename = "case1_final_network",
    #         prog = 'twopi',
    #         with_labels = True,
    #         font_weight = 'normal',
    #         font_size = 10,
    #         node_size = 120,
    #         node_color = 'blue',
    #         alpha = 0.6,
    #         arrowstyle = '->',
    #         arrowsize = 10
    #     )


    ##################### CASE 2 #######################
    # Initialize Case 2
    case2 = Case_2(data = raw_data)

    # Run Case 2 forwards
    case2.run(T = 14)

    if save_images:
        # Plot time series and save file in figures directory
        case2.save_entropy_time_series_plot(
            case2.topological_entropy_time_series,
            filename ='case2_topological_entropy_time_series',
            title = 'case2_topological_entropy_time_series',
            x_min = 0,
            x_max = 14,
            y_min = 0,
            y_max = 5,
            grid = True
        )

        case2.save_entropy_time_series_plot(
            case2.simple_entropy_time_series,
            filename ='case2_simple_entropy_time_series',
            title = 'case2_simple_entropy_time_series',
            x_min = 0,
            x_max = 14,
            y_min = 0,
            y_max = 5,
            grid = True
        )

        case2.save_entropy_time_series_plot(
            case2.target_value_entropy_time_series,
            filename ='case2_target_value_entropy_time_series_CRE',
            title = 'case2_target_value_entropy_time_series (CRE)',
            x_min = 0,
            x_max = 14,
            y_min = 0,
            y_max = None,
            grid = True
        )

        # Plot final network layout
        case2.save_final_network_plot(
            filename = "case2_final_network",
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



    # ############################ CASE 3 ###############################
    # # Initialize Case 3
    # case3_data = raw_data.calc_vols()
    #
    # case3 = Case_3(data = case3_data)
    #
    # # Run Case 3 forwards
    # case3.run(T = 14)
    # #print case3.network.nodes(data=True)
    #
    # if save_images:
    #     # Plot time series and save file in figures directory
    #     case3.save_entropy_time_series_plot(
    #         case3.topological_entropy_time_series,
    #         filename ='case3_topological_entropy_time_series',
    #         title = 'case3_topological_entropy_time_series',
    #         x_min = 0,
    #         x_max = 14,
    #         y_min = 0,
    #         y_max = 5,
    #         grid = True
    #     )
    #
    #     case3.save_entropy_time_series_plot(
    #         case3.simple_entropy_time_series,
    #         filename ='case3_simple_entropy_time_series',
    #         title = 'case3_simple_entropy_time_series',
    #         x_min = 0,
    #         x_max = 14,
    #         y_min = 0,
    #         y_max = 5,
    #         grid = True
    #     )
    #
    #     case3.save_entropy_time_series_plot(
    #         case3.target_value_entropy_time_series,
    #         filename ='case3_target_value_entropy_time_series_CRE',
    #         title = 'case3_target_value_entropy_time_series (CRE)',
    #         x_min = 0,
    #         x_max = 14,
    #         y_min = 0,
    #         y_max = None,
    #         grid = True
    #     )
    #
    #     # Plot final network layout
    #     case3.save_final_network_plot(
    #         filename = "case3_final_network",
    #         prog = 'twopi',
    #         with_labels = True,
    #         font_weight = 'normal',
    #         font_size = 10,
    #         node_size = 120,
    #         node_color = 'blue',
    #         alpha = 0.6,
    #         arrowstyle = '->',
    #         arrowsize = 10
    #     )


if __name__ == '__main__':
    main()
