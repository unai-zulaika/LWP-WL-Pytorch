import argparse
import time

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import data_processor as preprocess_dataset
from model import LWP_WL, LWP_WL_NO_CNN, LWP_WL_SIMPLE_CNN

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
parser.add_argument(
    '--dataset',
    default='all',
    choices=['all', 'airport', 'collaboration', 'congress', 'forum', 'usair'],
    help=
    'dataset to process: all (default) | airport | collaboration | congress | forum | usair'
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
    datasets = ['airport', 'collaboration', 'congress', 'forum']
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

    # mini batching
    batch_size = 32
    steps = 3000

    for s in range(int(x_train.shape[0] / 32)):
        optimizer.zero_grad()
        if s == int(x_train.shape[0] / 32) - 1:
            output = model(x_train[s * 32:, ...])
            loss_train = F.mse_loss(torch.flatten(output),
                                    y_train[s * 32:, ...])
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
