from utils.smiles_process import  smiles_to_fp
from model.eg_network import EG_MLP

from alg import egmcts
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import random
import logging
import argparse
import pickle
import time
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_target_molecules(dataset):
    if dataset == 'n1':
        target_mols = '../Data/PaRoutes/n1-targets.txt'
    elif dataset == 'n5':
        target_mols = '../Data/PaRoutes/n5-targets.txt'
    elif dataset == 'retro190':
        target_mols = '../Data/Retro190/retro190_targets.txt'
    elif dataset == 'drugbank':
        target_mols = '../Data/DrugBank/drugbank_targets.txt'
    elif dataset == 'literature30':
        target_mols = '../Data/Literature30/literature30_targets.txt'
    elif dataset == 'patent200':
        target_mols = '../Data/Patent200/patent200_targets.txt'
    with open(target_mols, 'r') as f:
        target_mols = [line.rstrip('\n') for line in f.readlines()]
    return target_mols


def prepare_starting_molecules(dataset):
    if dataset == 'n1':
        starting_mols = '../Data/PaRoutes/n1-stock.txt'
    elif dataset == 'n5':
        starting_mols = '../Data/PaRoutes/n5-stock.txt'
    else:
        starting_mols = '../Data/eMolecules.txt'
    with open(starting_mols, 'r') as f:
        starting_mols = [line.rstrip('\n') for line in f.readlines()]
    return starting_mols


def prepare_value():
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    model = EG_MLP(
            n_layers=1,
            fp_dim=2048,
            latent_dim=256,
            dropout_rate=0.1,
            device=device
    ).to(device)

    model_f = '../saved_model/EG_MCTS_default/value_pc.pt'
    logging.info('Loading Experience Guidance Network from %s' % model_f)

    saved_state_dict = torch.load(model_f, weights_only=True)
    current_state_dict = model.state_dict()

    for name, param in current_state_dict.items():
        if name in saved_state_dict:
            saved_param = saved_state_dict[name]
            if 'layers.0.weight' in name:
                param.data.copy_(saved_param[:, :2048])
            elif param.shape == saved_param.shape:
                param.data.copy_(saved_param)
            else:
                logging.warning(f"Skipping parameter {name} due to shape mismatch")
        else:
            logging.warning(f"Parameter {name} not found in z")

    model.load_state_dict(current_state_dict)
    model.eval()
    return model


def prepare_expand(ss_model):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    import sys; sys.path.append('..')
    if ss_model == 'default':
        from utils.prepare_methods import  prepare_mlp
        template_path = '../saved_model/EG_MCTS_default/template_rules.dat'
        model_path = '../saved_model/EG_MCTS_default/policy_model.ckpt'
        model = prepare_mlp(template_path, model_path)
    if ss_model == 'AZF':
        from ss_models.AZF import Model
        model = Model()
    if ss_model == 'LocalRetro':
        from ss_models.LocalRetro import Model
        model = Model(device)
    if ss_model == 'Chemformer':
        from ss_models.Chemformer import Model
        model = Model(device)
    if ss_model == 'ReactionT5':
        from ss_models.ReactionT5 import Model
        model = Model(device)
    return model


# eg_mcts
def prepare_egmcts_planner(args, one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(args, x)
    plan_handle = lambda x: egmcts(
        target_mol=x,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir
    )
    return plan_handle


def value_fn(mol):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    mol_fp = smiles_to_fp(mol, fp_dim=2048).reshape(1,-1)
    fp = mol_fp
    fp = torch.FloatTensor(fp).to(device)
    v = VALUE_MODEL(fp).item()
    return v
       
    
def EG_MCTS_plan(cfg):
    starting_mols = prepare_starting_molecules(cfg.dataset)

    targets = prepare_target_molecules(cfg.dataset)
    targets = targets[int(cfg.start):int(cfg.end)]

    one_step = prepare_expand(cfg.ss_model)

    plan_handle = prepare_egmcts_planner(
        args = cfg, 
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=cfg.expansion_topk,
        iterations=cfg.iterations,
        viz=False,
        viz_dir=None
    )
    # all results
    result = {
        'success': [],
        'time': [],
        'iter': [],
        'routes': [],
        'route_lens': [],
    }
    

    for target_mol in tqdm(targets):
        t0 = time.time()
        try:
            succ, route, msg, experience = plan_handle(target_mol)
            assert (time.time() - t0) < 600
            result['success'].append(succ)
            result['time'].append(time.time() - t0)
            result['iter'].append(msg[0])
            result['routes'].append(route)
            if succ:
                result['route_lens'].append(route.length)
            else:
                result['route_lens'].append(None)
            
        except:
            result['success'].append(False)
            result['time'].append(600)
            result['iter'].append(200)
            result['routes'].append(None)
            result['route_lens'].append(32)


    successes = np.mean(result['success'])
    print('Success: ', round(np.mean(successes), 4))
    
    save_path = f'{cfg.save_dir}/pred_{cfg.dataset}_{cfg.start}_{cfg.end}.pkl'
    print(save_path)
    f = open(save_path, 'wb')
    pickle.dump(result, f)
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--expansion_topk', type=int, default=10)

    parser.add_argument('--dataset', '-d')
    parser.add_argument('--ss_model', '-ss')

    parser.add_argument('--start', '-s', default=0)
    parser.add_argument('--end', '-e', default=10)

    args = parser.parse_args()
    args.alg = 'MCTS'
    
    set_seed(42)
    
    VALUE_MODEL = prepare_value()
    ss_model = args.ss_model
    save_dir = f"prediction/MCTS_{ss_model}"
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    EG_MCTS_plan(args)


    
