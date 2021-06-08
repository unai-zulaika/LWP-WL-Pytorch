import argparse
import time

import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
import torch.optim as optim

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph as sg

from gensim.models import Word2Vec

import data_processor as preprocess_dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.sparse import identity
from model import LWP_WL, LWP_WL_NO_CNN, LWP_WL_SIMPLE_CNN, Node2Vec
from pygcn.models import GCN, InnerProductDecoder

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def node2vec():
    print('Training Node2Vec mode!')

    # initialize results arrays
    total_mse = np.zeros(args.exp_number)

    total_pcc = np.zeros(args.exp_number)
    total_mae = np.zeros(args.exp_number)
    mse_datasets = {}
    std_datasets = {}
    pcc_datasets = {}
    pcc_std_datasets = {}
    mae_datasets = {}
    mae_std_datasets = {}

    t_total = time.time()

    if args.dataset == 'all':
        datasets = [
            'airport', 'collaboration', 'congress', 'forum', 'geom', 'astro'
        ]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        for exp_number in range(args.exp_number):
            print("%s: experiment number %d" % (dataset, exp_number + 1))

            data = preprocess_dataset.clean_data(dataset)
            if dataset != 'usair':
                data['weights'] = preprocessing.normalize([data['weights']])[0]

            # random split of data
            data_train, data_test = train_test_split(data, test_size=0.2)
            data_train, data_val = train_test_split(data_train, test_size=0.08)

            data = data.reset_index()
            data_train = data_train.reset_index()
            data_val = data_val.reset_index()
            data_test = data_test.reset_index()

            G = preprocess_dataset.create_graph_gcn(dataset, data, data_train)
            val_G = preprocess_dataset.create_graph_gcn(
                dataset, data, data_val)
            test_G = preprocess_dataset.create_graph_gcn(
                dataset, data, data_test)

            nodes_len = len(G.nodes)
            node_ids_to_index = {}
            for i, node_id in enumerate(G.nodes):
                node_ids_to_index[node_id] = i

            train_A = nx.adjacency_matrix(G)
            val_A = nx.adjacency_matrix(val_G)
            test_A = nx.adjacency_matrix(test_G)

            train_labels = torch.FloatTensor(
                data_train['weights'].values).cuda()
            val_labels = torch.FloatTensor(data_val['weights'].values).cuda()
            test_labels = torch.FloatTensor(data_test['weights'].values).cuda()

            train_A = sparse_mx_to_torch_sparse_tensor(train_A).cuda()
            val_A = sparse_mx_to_torch_sparse_tensor(val_A).cuda()
            test_A = sparse_mx_to_torch_sparse_tensor(test_A).cuda()

            G = sg.from_networkx(G)
            rw = BiasedRandomWalk(G)
            weighted_walks = rw.run(
                nodes=G.nodes(),  # root nodes
                length=args.length,  # maximum length of a random walk
                n=args.n_size,  # number of random walks per root node
                p=args.p,  # Defines (unormalised) probability, 1/p, of returning to source node
                q=args.q,  # Defines (unormalised) probability, 1/q, for moving away from source node
                weighted=True,  # for weighted random walks
                seed=42,  # random seed fixed for reproducibility
            )
            print("Number of random walks: {}".format(len(weighted_walks)))
            weighted_model = Word2Vec(weighted_walks, vector_size=args.vector_size, window=5, min_count=0, sg=1, workers=4)
            weights = torch.FloatTensor(weighted_model.wv.vectors).cuda()

            ########################################

            train_n1 =  torch.tensor(data_train['A'].values).cuda()
            train_n2 =  torch.tensor(data_train['B'].values).cuda()

            train_n1_indices = torch.ones(train_n1.shape[0])
            for i, value in enumerate(train_n1):
                train_n1_indices[i] = node_ids_to_index[value.item()]
            train_n1_indices = train_n1_indices.cuda().long()

            train_n2_indices = torch.ones(train_n1.shape[0])
            for i, value in enumerate(train_n2):
                train_n2_indices[i] = node_ids_to_index[value.item()]
            train_n2_indices = train_n2_indices.cuda().long()

            ########################################

            val_n1 =  torch.tensor(data_val['A'].values).cuda()
            val_n2 =  torch.tensor(data_val['B'].values).cuda()

            val_n1_indices = torch.ones(val_n1.shape[0])
            for i, value in enumerate(val_n1):
                val_n1_indices[i] = node_ids_to_index[value.item()]
            val_n1_indices = val_n1_indices.cuda().long()

            val_n2_indices = torch.ones(val_n1.shape[0])
            for i, value in enumerate(val_n2):
                val_n2_indices[i] = node_ids_to_index[value.item()]
            val_n2_indices = val_n2_indices.cuda().long()

            ########################################
            
            test_n1 =  torch.tensor(data_test['A'].values).cuda()
            test_n2 =  torch.tensor(data_test['B'].values).cuda()

            test_n1_indices = torch.ones(test_n1.shape[0])
            for i, value in enumerate(test_n1):
                test_n1_indices[i] = node_ids_to_index[value.item()]
            test_n1_indices = test_n1_indices.cuda().long()

            test_n2_indices = torch.ones(test_n1.shape[0])
            for i, value in enumerate(test_n2):
                test_n2_indices[i] = node_ids_to_index[value.item()]
            test_n2_indices = test_n2_indices.cuda().long()

            ########################################

            model = Node2Vec(weights, 0.5)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            model.train()
            model = model.to(args.device)

            # train
            for epoch in range(args.epochs):
                t = time.time()
                model.train()
                optimizer.zero_grad()

                output = model(train_n1_indices, train_n2_indices)

                loss_train = F.mse_loss(output, train_labels)
                loss_train.backward()
                optimizer.step()

                # validation
                model.eval()
                output = model(val_n1_indices, val_n2_indices)
                loss_val = F.mse_loss(output, val_labels)

                if args.verbose:
                    print('Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'loss_val: {:.4f}'.format(loss_val.item()),
                          'time: {:.4f}s'.format(time.time() - t))

            # test
            model.eval()
            with torch.no_grad():
                output = model(test_n1_indices, test_n2_indices)
                
                loss_test = F.mse_loss(torch.flatten(output), test_labels)
                pcc_test = pearson_correlation(test_labels, output)
                mae_test = F.l1_loss(output, test_labels)
                print("Test set results:", "loss= {:.10f}".format(loss_test.item()),
                    "pcc= {:.10f}".format(pcc_test),
                    "mae= {:.10f}".format(mae_test.item()))

                total_mse[exp_number] = loss_test
                total_pcc[exp_number] = pcc_test
                total_mae[exp_number] = mae_test


        # results
        mse_datasets[dataset] = np.mean(total_mse)
        std_datasets[dataset] = np.std(total_mse)
        total_mse = np.zeros(args.exp_number)

        pcc_datasets[dataset] = np.mean(total_pcc[~np.isnan(total_pcc)])
        pcc_std_datasets[dataset] = np.std(total_pcc[~np.isnan(total_pcc)])
        total_pcc = np.zeros(args.exp_number)

        mae_datasets[dataset] = np.mean(total_mae)
        mae_std_datasets[dataset] = np.std(total_mae)
        total_mae = np.zeros(args.exp_number)

    for dataset in datasets:
        print("MSE %s: {:,f}".format(mse_datasets[dataset]) % dataset)
        print("MSE_STD %s: {:,f}".format(std_datasets[dataset]) % dataset)

        print("PCC %s: {:,f}".format(pcc_datasets[dataset]) % dataset)
        print("PCC_STD %s: {:,f}".format(pcc_std_datasets[dataset]) % dataset)

        print("MAE %s: {:,f}".format(mae_datasets[dataset]) % dataset)
        print("MAE_STD %s: {:,f}".format(mae_std_datasets[dataset]) % dataset)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    exit()

def gcn():
    print('Training GCN mode!')

    # initialize results arrays
    total_mse = np.zeros(args.exp_number)

    total_pcc = np.zeros(args.exp_number)
    total_mae = np.zeros(args.exp_number)
    mse_datasets = {}
    std_datasets = {}
    pcc_datasets = {}
    pcc_std_datasets = {}
    mae_datasets = {}
    mae_std_datasets = {}

    t_total = time.time()

    if args.dataset == 'all':
        datasets = [
            'airport', 'collaboration', 'congress', 'forum', 'geom', 'astro'
        ]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        for exp_number in range(args.exp_number):
            print("%s: experiment number %d" % (dataset, exp_number + 1))

            data = preprocess_dataset.clean_data(dataset)
            if dataset != 'usair':
                data['weights'] = preprocessing.normalize([data['weights']])[0]

            # random split of data
            data_train, data_test = train_test_split(data, test_size=0.2)
            data_train, data_val = train_test_split(data_train, test_size=0.08)

            data = data.reset_index()
            data_train = data_train.reset_index()
            data_val = data_val.reset_index()
            data_test = data_test.reset_index()

            G = preprocess_dataset.create_graph_gcn(dataset, data, data_train)
            val_G = preprocess_dataset.create_graph_gcn(
                dataset, data, data_val)
            test_G = preprocess_dataset.create_graph_gcn(
                dataset, data, data_test)

            nodes_len = len(G.nodes)
            node_ids_to_index = {}
            for i, node_id in enumerate(G.nodes):
                node_ids_to_index[node_id] = i

            train_A = nx.adjacency_matrix(G)
            val_A = nx.adjacency_matrix(val_G)
            test_A = nx.adjacency_matrix(test_G)

            # identity if same for all
            I = identity(G.number_of_nodes(), dtype='int8', format='csr')
            features = torch.FloatTensor(np.array(I.todense())).cuda()

            train_labels = torch.FloatTensor(
                data_train['weights'].values).cuda()
            val_labels = torch.FloatTensor(data_val['weights'].values).cuda()
            test_labels = torch.FloatTensor(data_test['weights'].values).cuda()

            train_A = sparse_mx_to_torch_sparse_tensor(train_A).cuda()
            val_A = sparse_mx_to_torch_sparse_tensor(val_A).cuda()
            test_A = sparse_mx_to_torch_sparse_tensor(test_A).cuda()

            model = GCN(nfeat=nodes_len, nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            model.train()
            model = model.to(args.device)

            # train
            for epoch in range(args.epochs):
                t = time.time()
                model.train()
                optimizer.zero_grad()

                output = model(features, train_A,
                               torch.tensor(data_train['A'].values).cuda(),
                               torch.tensor(data_train['B'].values).cuda(),
                               node_ids_to_index)

                loss_train = F.mse_loss(output, train_labels)
                loss_train.backward()
                optimizer.step()

                # validation
                model.eval()
                output = model(features, val_A,
                               torch.tensor(data_val['A'].values).cuda(),
                               torch.tensor(data_val['B'].values).cuda(),
                               node_ids_to_index)
                loss_val = F.mse_loss(output, val_labels)

                if args.verbose:
                    print('Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'loss_val: {:.4f}'.format(loss_val.item()),
                          'time: {:.4f}s'.format(time.time() - t))

            # test
            model.eval()
            with torch.no_grad():
                output = model(features, test_A,
                               torch.tensor(data_test['A'].values).cuda(),
                               torch.tensor(data_test['B'].values).cuda(),
                               node_ids_to_index)
                
                loss_test = F.mse_loss(torch.flatten(output), test_labels)
                pcc_test = pearson_correlation(test_labels, output)
                mae_test = F.l1_loss(output, test_labels)
                print("Test set results:", "loss= {:.10f}".format(loss_test.item()),
                    "pcc= {:.10f}".format(pcc_test),
                    "mae= {:.10f}".format(mae_test.item()))

                total_mse[exp_number] = loss_test
                total_pcc[exp_number] = pcc_test
                total_mae[exp_number] = mae_test


        # results
        mse_datasets[dataset] = np.mean(total_mse)
        std_datasets[dataset] = np.std(total_mse)
        total_mse = np.zeros(args.exp_number)

        pcc_datasets[dataset] = np.mean(total_pcc[~np.isnan(total_pcc)])
        pcc_std_datasets[dataset] = np.std(total_pcc[~np.isnan(total_pcc)])
        total_pcc = np.zeros(args.exp_number)

        mae_datasets[dataset] = np.mean(total_mae)
        mae_std_datasets[dataset] = np.std(total_mae)
        total_mae = np.zeros(args.exp_number)

    for dataset in datasets:
        print("MSE %s: {:,f}".format(mse_datasets[dataset]) % dataset)
        print("MSE_STD %s: {:,f}".format(std_datasets[dataset]) % dataset)

        print("PCC %s: {:,f}".format(pcc_datasets[dataset]) % dataset)
        print("PCC_STD %s: {:,f}".format(pcc_std_datasets[dataset]) % dataset)

        print("MAE %s: {:,f}".format(mae_datasets[dataset]) % dataset)
        print("MAE_STD %s: {:,f}".format(mae_std_datasets[dataset]) % dataset)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    exit()


parser = argparse.ArgumentParser(description='Main file')
parser.add_argument('--disable-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='verbose training')
parser.add_argument('--unweighted',
                    action='store_true',
                    default=False,
                    help='do not compute weigths in adjacency matrix')
parser.add_argument('--no-cnn',
                    action='store_true',
                    default=False,
                    help='disables CNN layer on model')
parser.add_argument('--simple-cnn',
                    action='store_true',
                    default=False,
                    help='Simple CNN layer without special filters on model')
parser.add_argument('--random',
                    action='store_true',
                    default=False,
                    help='enables random node labeling (no W-WL)')
parser.add_argument('--gcn',
                    action='store_true',
                    default=False,
                    help='Use GCN model')
parser.add_argument('--node2vec',
                    action='store_true',
                    default=False,
                    help='Use node2vec model')
parser.add_argument(
    '--dataset',
    default='all',
    choices=[
        'all', 'airport', 'collaboration', 'congress', 'forum', 'usair',
        'astro', 'geom'
    ],
    help=
    'dataset to process: all (default) | airport | collaboration | congress | forum | usair | astro | geom'
)
parser.add_argument('--dir_data',
                    default='./data/',
                    help='directory where data is located')
parser.add_argument('--epochs',
                    '-e',
                    default=10,
                    type=int,
                    help='Number of epochs')
parser.add_argument('--k',
                    '-k',
                    default=10,
                    type=int,
                    help='size of the subgraph, K')
parser.add_argument('--exp_number',
                    '-en',
                    default=25,
                    type=int,
                    help='Number of experiments')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='initial learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-2,
                    help='weight decay (L2 loss on parameters)')
######################## GCN ########################
parser.add_argument('--nhid',
                    default=32,
                    type=int,
                    help='Number of hidden neurons, ONLY FOR GCN')
parser.add_argument('--nclass',
                    default=8,
                    type=int,
                    help='size of resulting node embedding, ONLY FOR GCN')
parser.add_argument('--dropout',
                    type=float,
                    default=0.5,
                    help='Dropout value')

######################## NODE2VEC ########################
parser.add_argument('--length',
                    default=50,
                    type=int,
                    help='maximum length of a random walk, ONLY FOR Node2Vec')
parser.add_argument('--n_size',
                    default=10,
                    type=int,
                    help='number of random walks per root node, ONLY FOR Node2Vec')
parser.add_argument('--vector_size',
                    default=8,
                    type=int,
                    help='size of resulting embedding, ONLY FOR Node2Vec')
parser.add_argument('--p',
                    type=float,
                    default=0.5,
                    help='Defines (unormalised) probability, 1/p, of returning to source node, ONLY FOR Node2Vec')
parser.add_argument('--q',
                    type=float,
                    default=2.0,
                    help='Defines (unormalised) probability, 1/q, for moving away from source node, ONLY FOR Node2Vec')

args = parser.parse_args()

# setup cpu or gpu
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# Set random seed
randseed = 3

np.random.seed(randseed)
torch.manual_seed(randseed)

if torch.cuda.is_available:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(randseed)

def initialize_model():
    # model and optimizer
    if args.no_cnn:
        model = LWP_WL_NO_CNN(args.k)
    elif args.simple_cnn:
        model = LWP_WL_SIMPLE_CNN(args.k)
    else:
        model = LWP_WL(args.k)

    if args.gcn:
        gcn()
    
    if args.node2vec:
        node2vec()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return model, optimizer


# initialize results arrays
total_mse = np.zeros(args.exp_number)

total_pcc = np.zeros(args.exp_number)
total_mae = np.zeros(args.exp_number)
mse_datasets = {}
std_datasets = {}
pcc_datasets = {}
pcc_std_datasets = {}
mae_datasets = {}
mae_std_datasets = {}

if args.dataset == 'all':
    datasets = [
        'airport', 'collaboration', 'congress', 'forum', 'geom', 'astro'
    ]
else:
    datasets = [args.dataset]

t_total = time.time()


def pearson_correlation(numbers_x, numbers_y):
    mean_x = torch.mean(numbers_x)
    mean_y = torch.mean(numbers_y)
    subtracted_mean_x = numbers_x - mean_x
    subtracted_mean_y = numbers_y - mean_y
    x_times_y = subtracted_mean_x * subtracted_mean_y
    x_squared = subtracted_mean_x**2
    y_squared = subtracted_mean_y**2
    return torch.sum(x_times_y) / torch.sqrt(
        torch.sum(x_squared) * torch.sum(y_squared))


def train(epoch, x_train, y_train, x_val, y_val, optimizer, model):
    t = time.time()
    model.train()
    model.double()

    for s in range(int(x_train.shape[0] / 32)):
        optimizer.zero_grad()
        if s == int(x_train.shape[0] / 32) - 1:
            output = model(x_train[s * 32:, ...])
            loss_train = F.mse_loss(torch.flatten(output), y_train[s * 32:,
                                                                   ...])
        else:
            output = model(x_train[s * 32:(s + 1) * 32, ...])
            loss_train = F.mse_loss(torch.flatten(output),
                                    y_train[s * 32:(s + 1) * 32, ...])
        loss_train.backward()
        optimizer.step()

    # validation
    model.eval()
    output = model(x_val)
    loss_val = F.mse_loss(torch.flatten(output), y_val)

    if args.verbose:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def test(x_test, y_test, total_mse, total_pcc, total_mae, exp_number, model):
    model.eval()
    with torch.no_grad():
        output = model(x_test)
        loss_test = F.mse_loss(torch.flatten(output), y_test)

        pcc_test = pearson_correlation(y_test, torch.flatten(output))
        mae_test = F.l1_loss(torch.flatten(output), y_test)
        print("Test set results:", "loss= {:.10f}".format(loss_test.item()),
              "pcc= {:.10f}".format(pcc_test),
              "mae= {:.10f}".format(mae_test.item()))

        total_mse[exp_number] = loss_test
        total_pcc[exp_number] = pcc_test
        total_mae[exp_number] = mae_test


# for each dataset we will do args.exp_numb experiments
for dataset in datasets:
    for exp_number in range(args.exp_number):
        print("%s: experiment number %d" % (dataset, exp_number + 1))
        model, optimizer = initialize_model()
        model = model.to(args.device)

        # random data splits
        x_train, y_train, x_val, y_val, x_test, y_test = preprocess_dataset.generate_data(
            dataset, args.random, args.k, args.unweighted)
        x_train = np.expand_dims(x_train, axis=1)
        x_val = np.expand_dims(x_val, axis=1)
        x_test = np.expand_dims(x_test, axis=1)

        x_train = torch.from_numpy(x_train).to(args.device)
        y_train = torch.from_numpy(y_train).to(args.device)
        x_val = torch.from_numpy(x_val).to(args.device)
        y_val = torch.from_numpy(y_val).to(args.device)
        x_test = torch.from_numpy(x_test).to(args.device)
        y_test = torch.from_numpy(y_test).to(args.device)

        for epoch in range(args.epochs):
            train(epoch, x_train, y_train, x_val, y_val, optimizer, model)
        test(x_test, y_test, total_mse, total_pcc, total_mae, exp_number,
             model)

    mse_datasets[dataset] = np.mean(total_mse)
    std_datasets[dataset] = np.std(total_mse)
    total_mse = np.zeros(args.exp_number)

    pcc_datasets[dataset] = np.mean(total_pcc[~np.isnan(total_pcc)])
    pcc_std_datasets[dataset] = np.std(total_pcc[~np.isnan(total_pcc)])
    total_pcc = np.zeros(args.exp_number)

    mae_datasets[dataset] = np.mean(total_mae)
    mae_std_datasets[dataset] = np.std(total_mae)
    total_mae = np.zeros(args.exp_number)

for dataset in datasets:
    print("MSE %s: {:,f}".format(mse_datasets[dataset]) % dataset)
    print("MSE_STD %s: {:,f}".format(std_datasets[dataset]) % dataset)

    print("PCC %s: {:,f}".format(pcc_datasets[dataset]) % dataset)
    print("PCC_STD %s: {:,f}".format(pcc_std_datasets[dataset]) % dataset)

    print("MAE %s: {:,f}".format(mae_datasets[dataset]) % dataset)
    print("MAE_STD %s: {:,f}".format(mae_std_datasets[dataset]) % dataset)

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
