from valueEnsemble import ValueEnsemble
from contextlib import contextmanager
from tqdm import tqdm
from value_functions import value_fn_default as value_fn

import os
import time
import pickle
import signal
import argparse
import random
import logging
import torch

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*", category=DeprecationWarning)


class TimeoutException(Exception): pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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

        
class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, parent=None, cpuct=1.5):
        self.state = state
        self.cost = cost
        self.h = h
        self.prior = prior
        self.visited_time = 0
        self.is_expanded = False
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.child_illegal = np.array([])
        if parent is not None:
            self.g = self.parent.g + cost
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.f_mean_path = []

    def child_N(self):
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats):
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        best_move = np.argmax(action_score)
        return best_move


def prepare_value(model_f, device):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device, weights_only=True))
    model.eval()
    return model

def prepare_expand(ss_model):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    import sys; sys.path.append('..')
    if ss_model == 'default':
        from ss_models.default import MLPModel
        model_path = '../saved_model/MEEA_default/policy_model.ckpt'
        template_path = '../saved_model/MEEA_default/template_rules.dat'
        model = MLPModel(model_path, template_path , device=device)
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



class MCTS_A:
    def __init__(self, args, target_mol, known_mols, value_model, expand_fn, device, simulations, cpuct):
        self.args = args
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.cpuct = cpuct
        root_value = value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=self.cpuct)
        self.open = [self.root]
        self.visited_policy = {}
        self.visited_state = []
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.opening_size = simulations
        self.iterations = 0
    
    def select_a_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best_move = current.select_child(self.min_max_stats)
            current = current.children[best_move]

    def select(self):
        openings = [self.select_a_leaf() for _ in range(self.opening_size)]
        stats = [opening.f for opening in openings]
        index = np.argmin(stats)
        return openings[index]

    def expand(self, node):
        node.is_expanded = True
        expanded_mol_index = 0
        expanded_mol = node.state[expanded_mol_index]
        if expanded_mol in self.visited_policy.keys():
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.expand_fn.run(args, expanded_mol)
            self.iterations += 1
            if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None

        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            node.child_illegal = np.array([0] * len(expanded_policy['scores']))
            for i in range(len(expanded_policy['reactants'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[: expanded_mol_index] + node.state[expanded_mol_index + 1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))
  
                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), parent=node, cpuct=self.cpuct)
                    return True, child
                else:
                    h = value_fn(self.value_model, reactant, self.device)
                    child = Node(reactant, h, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), parent=node, cpuct=self.cpuct)

                    if '.'.join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
        return False, None

    def update(self, node):
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current is not None:
            current.f_mean_path.append(stat)
            current = current.parent

    def search(self, times):
        success, node = False, None
        while self.iterations < times and not success and (not np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0):
            expand_node = self.select()
            if '.'.join(expand_node.state) in self.visited_state:
                expand_node.parent.child_illegal[expand_node.fmove] = 1000
                back_check_node = expand_node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
                continue
            else:
                self.visited_state.append('.'.join(expand_node.state))
                success, node = self.expand(expand_node)
                self.update(expand_node)
            if self.visited_policy[self.target_mol] is None:
                return False, None, times, self.visited_policy[self.target_mol]
        return success, node, self.iterations, self.visited_policy[self.target_mol] 

    def vis_synthetic_path(self, node):
        if node is None:
            return []
        reaction_path = []
        current = node
        while current is not None:
            reaction_path.append(current.reaction)
            current = current.parent
        return reaction_path[::-1]

def play(args, mols, known_mols, value_model, expand_fn, device, simulations, cpuct, times, save_path):
    routes = []
    successes = []
    depths = []
    counts = []
    visited_policies = []
    search_times = []

    for mol in tqdm(mols):
        start = time.time()
        if True:
            with time_limit(600):
                player = MCTS_A(args, mol, known_mols, value_model, expand_fn, device, simulations, cpuct)
                success, node, count, visited_policy = player.search(times)
                route = player.vis_synthetic_path(node)
                
        # except:
        #     success = False
        #     route = [None]
        #     visited_policy = [None]
                
        end = time.time()
        search_time = end - start
        search_times.append(search_time)

        routes.append(route)
        successes.append(success)
        if success:
            depths.append(node.depth)
            counts.append(count)
            visited_policies.append(visited_policy)
        else:
            depths.append(32)
            counts.append(-1)
            visited_policies.append(visited_policy)
            
        print(success)
        
    result = {
        'success': successes,
        'time': search_times,
        'iter': counts,
        'routes': routes,
        'route_lens': depths,
    }

    print('Success: ', round(np.mean(successes), 4))
    pickle.dump(result, open(save_path, 'wb'), protocol=4)
    print(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s')
    parser.add_argument('--end', '-e')
    parser.add_argument('--ss_model', '-ss', choices=['default', 'AZF', 'LocalRetro', 'Chemformer', 'ReactionT5'])
    parser.add_argument('--dataset', '-d', choices=['n1', 'n5', 'retro190', 'drugbank', 'literature30', 'patent200'])
    args = parser.parse_args()
    args.alg = 'MEEA'
    
    set_seed(42)
    
    targets = []
    ss_model = args.ss_model
    dataset = args.dataset
    simulations = 100
    cpuct = 4.0
    
    targets = prepare_target_molecules(dataset)
    known_mols = prepare_starting_molecules(dataset)

    one_step = prepare_expand(ss_model)
    
    model_f = '../saved_model/MEEA_default/value_pc.pt'
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    value_model = prepare_value(model_f, device)
    
    
    save_dir = f"prediction/MEEA_{ss_model}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/pred_{dataset}_{args.start}_{args.end}.pkl'
    
    for iters in [200]:
        play(args, targets[int(args.start): int(args.end)], known_mols, value_model, one_step, device, simulations, cpuct, iters, save_path)
