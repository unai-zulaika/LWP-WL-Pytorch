
from random import choice, shuffle
import os
import sys

import numpy as np
import networkx as nx
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

path = os.path.abspath(os.path.dirname(__file__))
DATASETS_PATH = os.path.join(path, 'data/raw/')


def clean_data(dataset):

    data = pd.read_csv(DATASETS_PATH + dataset + '.txt',
                       sep='\t',
                       names=['A', 'B', 'weights'])

    data = data[data.A != data.B]
    
    return data


def create_graph_gcn(dataset, data, data_train):
    G = nx.Graph()
    
    if not isinstance(data, pd.DataFrame):
        data = clean_data(dataset)

        if dataset != 'usair':
            data['weights'] = preprocessing.normalize([data['weights']])[0]   

    for index, row in data_train.iterrows():
        node_a = int(row['A'])
        node_b = int(row['B'])

        G.add_node(node_a)
        G.add_node(node_b)
        G.add_edge(node_a, node_b, weight=row['weights'])
    
    for index, row in data.iterrows():
        node_a = int(row['A'])
        node_b = int(row['B'])

        if node_a not in G:
            G.add_node(node_a)
        if node_b not in G:    
            G.add_node(node_b)     

    return G  


def create_graph(dataset, data=0):
    G = nx.Graph()
    
    if not isinstance(data, pd.DataFrame):
        data = clean_data(dataset)

        if dataset != 'usair':
            data['weights'] = preprocessing.normalize([data['weights']])[0]

    for index, row in data.iterrows():
        node_a = int(row['A'])
        node_b = int(row['B'])

        G.add_node(node_a)
        G.add_node(node_b)
        G.add_edge(node_a, node_b, weight=float(row['weights']))

    return G


def clean_data_unweighted(dataset):

    data = pd.read_csv(DATASETS_PATH + dataset + '.txt',
                       sep='\t',
                       names=['A', 'B'],
                       usecols=[0, 1])

    data = data[data.A != data.B]

    return data


def create_graph_unweighted(dataset):
    G = nx.Graph()

    data = clean_data(dataset)

    for index, row in data.iterrows():
        node_a = int(row[0])
        node_b = int(row[1])

        G.add_node(node_a)
        G.add_node(node_b)
        G.add_edge(node_a, node_b)

    return G


def generate_data(dataset, random, K_DEPTH, unweighted):

    if not unweighted:
        G = create_graph(dataset)

        data = clean_data(dataset)
        if dataset != 'usair':
            data['weights'] = preprocessing.normalize([data['weights']])[0]

    else:
        G = create_graph_unweighted(dataset)

        data = clean_data_unweighted(dataset)

    # random split of data
    data_train, data_test = train_test_split(data, test_size=0.2)
    data_train, data_val = train_test_split(data_train, test_size=0.08)

    data_train = data_train.reset_index()
    data_val = data_val.reset_index()
    data_test = data_test.reset_index()

    x_train = np.zeros([len(data_train), K_DEPTH, K_DEPTH])
    y_train = np.zeros([len(data_train)])

    x_val = np.zeros([len(data_val), K_DEPTH, K_DEPTH])
    y_val = np.zeros([len(data_val)])

    x_test = np.zeros([len(data_test), K_DEPTH, K_DEPTH])
    y_test = np.zeros([len(data_test)])

    for index, row in data_train.iterrows():
        node_a = int(row[1])
        node_b = int(row[2])

        subgraph = get_subgraph(G, node_a, node_b, random, K_DEPTH)

        x_train[index, ...] = subgraph
        y_train[index] = row[3]

    for index, row in data_val.iterrows():
        node_a = int(row[1])
        node_b = int(row[2])

        subgraph = get_subgraph(G, node_a, node_b, random, K_DEPTH)

        x_val[index, ...] = subgraph
        y_val[index] = row[3]

    for index, row in data_test.iterrows():
        node_a = int(row[1])
        node_b = int(row[2])

        subgraph = get_subgraph(G, node_a, node_b, random, K_DEPTH)

        x_test[index, ...] = subgraph
        y_test[index] = row[3]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_subgraph(G, node_a, node_b, random, K):
    """ Create subgraph for link prediction targets """

    NODE_SUBGRAPH_SIZE = int(K / 2)
    SUBGRAPH_SIZE = K
    subgraph_nodes = [node_a]

    index = 0
    while (len(subgraph_nodes) < NODE_SUBGRAPH_SIZE):
        for n in G.neighbors(subgraph_nodes[index]):
            if len(subgraph_nodes) == NODE_SUBGRAPH_SIZE:
                break
            if n != node_b and n not in subgraph_nodes:
                subgraph_nodes.append(n)
        index += 1

        if (index == len(subgraph_nodes)):
            while (len(subgraph_nodes) < NODE_SUBGRAPH_SIZE):
                random_node = choice(list(G))
                if random_node not in subgraph_nodes and random_node != node_b:
                    subgraph_nodes.append(random_node)

    subgraph_nodes.append(node_b)
    index = round(K / 2)
    while (len(subgraph_nodes) < SUBGRAPH_SIZE):
        for n in G.neighbors(subgraph_nodes[index]):
            if len(subgraph_nodes) == SUBGRAPH_SIZE:
                break
            if n not in subgraph_nodes:
                subgraph_nodes.append(n)
        index += 1

        if (index == len(subgraph_nodes)):
            while (len(subgraph_nodes) < SUBGRAPH_SIZE):
                random_node = choice(list(G))
                if random_node not in subgraph_nodes:
                    subgraph_nodes.append(random_node)

    subgraph = G.subgraph(subgraph_nodes)

    return label_nodes_wl(subgraph, subgraph_nodes, node_a, node_b, random)


def label_nodes_wl(G, nodelist, node_a, node_b, random):
    """ Order nodes from subgraph with WL algorithm """

    nodelist_ordered = []
    nodelist_initial_etiqs = generate_initial_etiqs(G, nodelist, node_a,
                                                    node_b)
    nodelist_full_etiqs = generate_full_etiqs(G, nodelist,
                                              nodelist_initial_etiqs, node_a,
                                              node_b)

    while (len(nodelist_ordered) < len(nodelist_full_etiqs)):
        next_node = choice(list(nodelist_full_etiqs))
        while (next_node in nodelist_ordered):
            next_node = choice(list(nodelist_full_etiqs))

        next_node_list = nodelist_full_etiqs[next_node]

        for key in nodelist_full_etiqs:
            if key not in nodelist_ordered:
                if not wl_list_comparison_max(next_node_list,
                                              nodelist_full_etiqs[key]):
                    next_node_list = nodelist_full_etiqs[key]
                    next_node = key

        nodelist_ordered.append(next_node)
    if random:
        shuffle(nodelist_ordered)
    matrix = nx.to_numpy_matrix(G.subgraph(nodelist_ordered),
                                nodelist=nodelist_ordered)
    matrix[0, 1] = -1
    matrix[1, 0] = -1
    return matrix


def generate_initial_etiqs(G, nodelist, node_a, node_b):
    """ Generate etiquettes for every node for WL algorithm """

    nodelist_initial_etiqs = {}

    for node in nodelist:
        try:
            distance_to_a = nx.shortest_path_length(G, node_a, node, 'weight')
            distance_to_b = nx.shortest_path_length(G, node_b, node, 'weight')

            nodelist_initial_etiqs[node] = distance_to_a + distance_to_b

        except nx.exception.NetworkXNoPath:
            nodelist_initial_etiqs[node] = sys.float_info.min

    nodelist_initial_etiqs[node_a] = sys.float_info.max
    nodelist_initial_etiqs[node_b] = sys.float_info.max

    return nodelist_initial_etiqs


def generate_full_etiqs(G, nodelist, nodelist_initial_etiqs, node_a, node_b):
    """ Generate etiquettes for every node and its neighbours for WL
    algorithm """
    nodelist_full_etiqs = {}

    for node in nodelist:
        nodelist_full_etiqs[node] = [nodelist_initial_etiqs[node]]
        for neighbour in G.neighbors(node):
            nodelist_full_etiqs[node].append(nodelist_initial_etiqs[neighbour])

        nodelist_full_etiqs[node][1:] = sorted(nodelist_full_etiqs[node][1:],
                                               reverse=True)

    return nodelist_full_etiqs


def wl_list_comparison_max(list_a, list_b):
    """ Compare two lists to get the best one by WL algorithm """
    for index, value in enumerate(list_a, 0):
        if index >= len(list_b):
            return 1
        if value > list_b[index]:
            return 1
        elif value < list_b[index]:
            return 0

    return 1


def wl_list_comparison_min(list_a, list_b):
    """ Compare two lists to get the best one by WL algorithm """
    for index, value in enumerate(list_a, 0):
        if index >= len(list_b):
            return 1
        if value > list_b[index]:
            return 0
        elif value < list_b[index]:
            return 1

    return 1
