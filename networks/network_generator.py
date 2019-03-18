
import sys
sys.path.append('../')
import networkx as nx
from data_sources.data_interpreter import Case_Data

## TO DO #####
'''
- Link the data status to the presence of data in add_attributes
- work out the calculation of the volumes from the variables using the data_interpreter
- implement entropy measures
- implement visualizations
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

    def init_network(self):
        G = nx.DiGraph()
        G.add_node(self.target_node, add_attributes('avg_vol',float(), 0))
        return G



class Case_1(Knowledge_Network):
    # This case computes the averages for each varaible, then multiplies them
    # together to get the average volume.

    def __init__(self, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)


class Case_2(Knowledge_Network):
    # This case computes a volume for each ship, then takes the averages of the
    # volumes.

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def grow(self):
        new_node_label = max(self.network.nodes()) + 1
        self.network.add_node(new_node_label, add_attributes('V'+str(self.time_step),0.0,0))
        self.network.add_edge(new_node_label, 0)
        self.network.add_node(new_node_label+1, add_attributes('L'+str(self.time_step),self.data.data[self.time_step]['L'],1))
        self.network.add_node(new_node_label+2, add_attributes('B'+str(self.time_step),self.data.data[self.time_step]['B'],1))
        self.network.add_node(new_node_label+3, add_attributes('T'+str(self.time_step),self.data.data[self.time_step]['T'],1))
        self.network.add_node(new_node_label+4, add_attributes('Cb'+str(self.time_step),self.data.data[self.time_step]['Cb'],1))
        self.network.add_edge(new_node_label+1, new_node_label)
        self.network.add_edge(new_node_label+2, new_node_label)
        self.network.add_edge(new_node_label+3, new_node_label)
        self.network.add_edge(new_node_label+4, new_node_label)


class Case_3(Knowledge_Network):
    # This case only imports the volumes associated with each ship, without
    # the associated variable values.

    def __init__(self, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)

############## Global functions ############
def add_attributes(name = '', value = float(), status=0):
    out_dict = {
    'name': name,
    'val': value,
    'data_status': status
    }
    return out_dict


############################################
def main():
    # Import the data
    data = Case_Data('../data_sources\\case_data.csv')

    case2 = Case_2(data = data)
    #print case2.data.data
    for i in range(1,15):
        case2.time_step += 1
        case2.grow()
        print case2.network.nodes(data = True)
        print ""
        #print case2.network.edges()


    #print test.network.nodes(data=True)
    #print max(test.network.nodes())

if __name__ == '__main__':
    main()
