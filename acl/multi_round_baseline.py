import os
import sys
import argparse
from tqdm import tqdm
from copy import deepcopy, copy
from datetime import datetime
import gc
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=1
# os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_DISABLE_GIT"] = "true"
os.environ["WANDB_ERROR_REPORTING"] = "false"

import numpy as np
import random

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss

from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, CAL_Plugin
from avalanche.training.supervised import Replay, Naive
from avalanche.query_strategies import *
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.checkpoint import save_checkpoint
from data import get_scenario, get_statistics
from acl.deepal_utils import get_net, get_cl_strategy

torch.set_num_threads(16)
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['WANDB_START_METHOD']="thread"
os.environ['WANDB_DIR']=os.path.abspath('/data3/jhpark/')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
def main(args):
    seed_everything(args.seed)
    NUM_INIT_LB = args.nStart
    NUM_QUERY = args.nQuery
    NUM_ROUND = int((args.nEnd-NUM_INIT_LB)/args.nQuery)
    fname = f'/data3/jhpark/ckpts/{args.al_strategy}_{args.scenario}_{args.cl_strategy}_{datetime.now()}.pkl'

    device = torch.device(
        f"cuda"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    
    al_plugin = getattr(sys.modules[__name__], args.al_strategy)
    model = get_net(args.data)
    scenario = get_scenario(args.scenario, args.seed, args.cl_strategy)
    img_size = get_statistics(args.data)['img_size']

    plugin = [CAL_Plugin()]

    ### l2p and dualprompt does not use this optimizer & schedueler
    if args.data  == 'TinyImageNet':
        optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)
    else:
        optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.8)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min = 0.001)
    scheduler_plugin = LRSchedulerPlugin(scheduler, step_granularity="epoch", first_exp_only=False)

    plugin.append(scheduler_plugin)
    criterion = CrossEntropyLoss()

    configs = {
        'nEnd': args.nEnd, 
        'al_strategy': args.al_strategy,
        'cl_strategy': args.cl_strategy,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'scenario': args.scenario,
    }

    loggers = [InteractiveLogger()]
    if args.wandb:
        project_name = f'prev_{args.cl_strategy}_{args.scenario}_{args.mem_size}'
        loggers.append(
            WandBLogger(project_name=project_name, run_name=f'{args.al_strategy}_{args.cl_strategy}_{args.nEnd}_{args.mem_size}', config=configs)
        )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=args.n_classes, save_image=False, stream=True),
        loggers=loggers,
    )

    cl_strategy = get_cl_strategy(model, optimizer, criterion, plugin, eval_plugin, device, img_size, args)

    results = []
    # try:
    data_num = 0
    for exp in scenario.train_stream:
        n_pool = len(exp._dataset)
        data_num+=n_pool
        # lam = sigmoid(-(data_num-n_pool)/n_pool)
        lam = n_pool/data_num

        save_checkpoint(cl_strategy, fname)
        
        X = torch.stack([data[0] for data in tqdm(exp._dataset.with_transforms('eval'))])
        y = torch.tensor(exp._dataset.targets)

        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

        al_strategy = al_plugin(X, y, idxs_lb, cl_strategy, args, fname, lam=lam, exp=exp)
        print("Start of experience: ",exp.current_experience)
        print("Current Classes: ",exp.classes_in_this_experience)

        if NUM_INIT_LB == args.nEnd:
            al_strategy.train(exp, no_update=False, first=True)
            print('Training completed')
            print('Computing accuracy on the whole test set')
            results.append(cl_strategy.eval(scenario.test_stream[:exp.current_experience+1], num_workers=args.num_workers))
            continue
        else:
            al_strategy.train(exp, no_update=True, first=True)

        for rd in range(1, NUM_ROUND+1):

            print('Round {}'.format(rd), flush=True)
            output = al_strategy.query(NUM_QUERY, n_aug=args.n_aug, alpha=args.alpha)
            q_idxs = output
            idxs_lb[q_idxs] = True
            al_strategy.update(idxs_lb)

            print("Current Classes: ",exp.classes_in_this_experience)
            if rd == NUM_ROUND:
                al_strategy.train(exp, no_update=False)
            else:
                al_strategy.train(exp, no_update=True)


        print('Training completed')
        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream[:exp.current_experience+1], num_workers=args.num_workers))

    os.remove(fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=2023, help="Select zero-indexed cuda device. -1 to use CPU.")

    parser.add_argument("--mem_size", type=int, default=2000, help="Set memory replay size")
    parser.add_argument("--batch_size", type=int, default=128, help="Set memory replay size")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3, help="Set memory replay size")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--nStart', type=int, default=100)
    parser.add_argument("--nQuery", type=int, default=100)
    parser.add_argument('--nEnd', type=int, default=300)
    parser.add_argument('--nAll', action='store_true')
    parser.add_argument("--al_strategy", type=str, default='RandomSampling')
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cl_strategy", type=str, default='ER')
    parser.add_argument("--odin", action='store_true')

    parser.add_argument("--scenario", type=str, default='SplitCIFAR10')
    parser.add_argument("--data", type=str, default='CIFAR10')
    parser.add_argument("--n_classes", type=int, default=10)
    # parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    main(args)