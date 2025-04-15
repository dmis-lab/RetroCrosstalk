import torch
import logging
import time
from retro_star.common import smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger


import os
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from retro_star.alg import molstar

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


def prepare_expand(ss_model):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    import sys; sys.path.append('..')
    if ss_model == 'default':
        from retro_star.packages.mlp_retrosyn.mlp_retrosyn.mlp_inference import MLPModel
        model_path = '../saved_model/RETRO_STAR_default/policy_model.ckpt'
        template_path = '../saved_model/RETRO_STAR_default/template_rules.dat'
        model = MLPModel(model_path, template_path, device=device)
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


def prepare_molstar_planner(args, one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(args, x)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir
    )
    return plan_handle

def prepare_value(model_f, device):
    model = ValueMLP(
        n_layers=1,
        fp_dim=2048,
        latent_dim=128,
        dropout_rate=0.1,
        device=device
    ).to(device)
    model.load_state_dict(torch.load(model_f, weights_only=True))
    model.eval()
    return model

def value_fn(mol):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
    fp = torch.FloatTensor(fp).to(device)
    v = VALUE_MODEL(fp).item()
    return v

class RSPlanner:
    def __init__(self,
                 args,
                 expansion_topk,
                 iterations,
                 viz=False,
                 viz_dir='viz'):
        self.args = args
        setup_logger()

        starting_mols = prepare_starting_molecules(args.dataset)

        one_step = prepare_expand(args.ss_model)

        self.plan_handle = prepare_molstar_planner(
            args, 
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )

    def plan(self, target_mols, save_dir):
        successes = []
        iters = []
        routes = []
        route_lens = []
        search_times = []

        
        for i, target_mol in tqdm(enumerate(target_mols)):
            t0 = time.time()
            try:
                succ, msg = self.plan_handle(target_mol)
                search_time = time.time() - t0
                assert search_time < 600
            except:
                succ = False
                
            if succ:
                successes.append(succ)
                iters.append(msg[1])
                routes.append(msg[0].serialize())
                route_lens.append(msg[0].length)
                search_times.append(search_time)
                
            else:
                successes.append(succ)
                iters.append(200)
                routes.append(None)
                route_lens.append(None)
                search_times.append(600)
            print(succ)

            print(f'{i}th Success: ', round(np.mean(successes), 4))
            
        result = {
        'success': successes,
        'time': search_times,
        'iter': iters,
        'routes': routes,
        'route_lens': route_lens,
        }

        print('Success: ', round(np.mean(successes), 4))
        save_path = f'{save_dir}/pred_{self.args.dataset}_{self.args.start}_{self.args.end}.pkl'
        pickle.dump(result, open(save_path, 'wb'), protocol=4)
        print(save_path)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--ss_model', '-ss', choices=['default', 'AZF', 'LocalRetro', 'Chemformer', 'ReactionT5'])
    parser.add_argument('--dataset', '-d', choices=['n1', 'n5', 'retro190', 'drugbank', 'literature30', 'patent200'])
    cfg = parser.parse_args()
    cfg.alg = 'RETRO'

    targets = []
    ss_model = cfg.ss_model
    dataset = cfg.dataset
    model_f = '../saved_model/RETRO_STAR_default/value_pc.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    VALUE_MODEL = prepare_value(model_f, device)

    planner = RSPlanner(
        cfg, 
        iterations=200,
        expansion_topk=10,
    )

    target_mols = prepare_target_molecules(dataset)
    target_mols = target_mols[cfg.start: cfg.end]
    
    save_dir = f"prediction/RETRO_{ss_model}"
    os.makedirs(save_dir, exist_ok=True)
    
    planner.plan(target_mols, save_dir)

