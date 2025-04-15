"""
MCTS (Monte Carlo Tree Search) Synthesis Route Analysis

This script evaluates the feasibility and yield of predicted retrosynthesis routes from 
MCTS-based models. It processes reaction pathways, calculates success rates, feasibility 
scores, and yield predictions for various datasets.

Usage:
    python mcts_analysis.py --tree MCTS --ss_model MODEL_NAME --dataset DATASET_NAME
"""

import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import re
import os
import json
import argparse
import sys
from transformers import AutoTokenizer
from alg.syn_route import SynRoute

# Add parent directory to path for importing local modules
sys.path.append('..')
from ReactionFeasibilityModel.rfm.featurizers.featurizer import ReactionFeaturizer
from ReactionFeasibilityModel.rfm.models import ReactionGNN
from yield_pred_inference.inference import ReactionT5Yield

# =============================================================================
# Constants
# =============================================================================

DATASET_SIZES = {
    'n1': 10000,
    'n5': 10000,
    'retro190': 190,
    'drugbank': 8313,
    'literature30': 30,
    'patent200': 200,
}

RFM_CHECKPOINT = '../saved_model/uspto_rfm.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TIMEOUT = 600  # Default timeout value in seconds

# =============================================================================
# File and Reaction Parsing Functions
# =============================================================================

def sort_pred_files(files):
    """
    Sort prediction files based on numerical order in filenames.
    
    Args:
        files: List of filenames to sort
        
    Returns:
        Sorted list of filenames
    """
    def extract_numbers(filename):
        numbers = re.findall(r'(\d+)_(\d+)', filename)
        return tuple(map(int, numbers[0])) if numbers else (0, 0)
    return sorted(files, key=extract_numbers)

def split_in_place(s, delimiter1=".", delimiter2=">>"):
    """
    Split reaction SMILES into reactants and product.
    
    Args:
        s: Reaction SMILES string
        delimiter1: Delimiter between reactants (default: ".")
        delimiter2: Delimiter between reactants and product (default: ">>")
        
    Returns:
        Tuple of (reactants_list, product) or None if parsing fails
    """
    if delimiter2 in s:
        b, c = s.split(delimiter2)
        if delimiter1 in b:
            reactants = b.split(delimiter1)
        else:
            reactants = [b]
        return (reactants, c)
    return None

def convert_eg_mcts_route(routes):
    """
    Convert MCTS routes to reaction list format.
    
    Args:
        routes: Route object from MCTS algorithm
        
    Returns:
        List of (reactants, product) tuples or None if routes is None
    """
    if routes is None:
        return None
        
    route_paths = []
    for i in range(routes.length):
        reaction = SynRoute.serialize_reaction(routes, i)
        if '>>' in reaction:
            product, reactant = reaction.split('>>')
            reaction = reactant + '>>' + product
            route_paths.append(reaction)
            
    return [split_in_place(path.strip(), '.', '>>') for path in route_paths]

# =============================================================================
# Model Preparation Functions
# =============================================================================

def prepare_yield_model():
    """
    Initialize and prepare the yield prediction model.
    
    Returns:
        Loaded and initialized yield prediction model
    """
    model_path = '../yield_pred_inference/best_model.ckpt'
    tokenizer = AutoTokenizer.from_pretrained('../yield_pred_inference/tokenizer', return_tensors='pt')
    model = ReactionT5Yield(tokenizer, DEVICE)
    
    state = torch.load(model_path, map_location='cpu')
    adjusted_state = {k.replace("model.", "", 1): v for k, v in state['state_dict'].items() if k.startswith("model.")}
    model.load_state_dict(adjusted_state)
    
    return model.to(DEVICE).eval()

def predict_yield(model, product, reactants):
    """
    Predict yield for a single reaction.
    
    Args:
        model: Yield prediction model
        product: Product SMILES string
        reactants: List of reactant SMILES strings
        
    Returns:
        Predicted yield value
    """
    input_rxn = f'REACTANT:{".".join(reactants)}REAGENT: PRODUCT:{product}'
    inputs = model.tokenizer([input_rxn], return_tensors='pt').to(DEVICE)
    return model(inputs).item()

def calculate_yields(model, reactions_list):
    """
    Calculate yields for a list of reactions.
    
    Args:
        model: Yield prediction model
        reactions_list: List of (reactants, product) tuples
        
    Returns:
        List of predicted yield values
    """
    return [predict_yield(model, product, reactants) for reactants, product in reactions_list]

# =============================================================================
# Dataset Loading Functions
# =============================================================================

def load_dataset_info(dataset):
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


def standardize_pred_keys(pred_file):
    """
    Load prediction file and standardize key names.
    
    Key mappings:
    - 'succ' -> 'success'
    - 'route' -> 'routes'
    - 'route_lens', 'depth' -> 'depth'
    - 'iter', 'counts' -> 'model_calls'
    - 'cumulated_time' -> 'time'
    
    Args:
        pred_file: Path to prediction file
        
    Returns:
        Dictionary with standardized keys
    """
    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)
    
    # Standardize keys
    standardized = {}
    
    # Success
    standardized['success'] = pred.get('succ', pred.get('success', []))
    
    # Routes
    standardized['routes'] = pred.get('route', pred.get('routes', []))
    
    # Depth
    standardized['depth'] = pred.get('route_lens', pred.get('depth', []))
    
    # Model calls
    standardized['model_calls'] = pred.get('iter', pred.get('counts', []))
    
    # Time
    standardized['time'] = pred.get('cumulated_time', pred.get('time', []))
    
    return standardized

# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_predictions(pred_list, dataset, args, model_name):
    """
    Evaluate predictions and calculate statistics.
    
    Args:
        pred_list: List of prediction file paths
        dataset: Dataset name
        args: Command-line arguments
        model_name: Model name for output file naming
        
    Returns:
        None (results are printed and saved to a JSON file)
    """
    # Load dataset information
    target_mols = load_dataset_info(dataset)

    # Initialize models
    rfm_model = ReactionGNN(checkpoint_path=RFM_CHECKPOINT).eval().to(DEVICE)
    featurizer = ReactionFeaturizer()
    yield_model = prepare_yield_model()

    # Get total number of predictions
    total_predictions = 0
    for pred_file in pred_list:
        total_predictions += len(standardize_pred_keys(pred_file)['success'])
    assert total_predictions == DATASET_SIZES[dataset]
    # Initialize statistics with fixed size
    stats = {
        'success_rate': [0] * total_predictions,
        'depths': [0] * total_predictions,
        'model_calls': [0] * total_predictions,
        'search_times': [DEFAULT_TIMEOUT] * total_predictions,  # Initialize with timeout value
        'feasibility': [-1] * total_predictions,  # Initialize with -1 to distinguish from calculated values
        'yield_scores': [-1] * total_predictions,
        'total_reactions': 0,
        'total_rfm_score': [-1] * total_predictions
    }

    # Process predictions
    current_idx = 0
    for pred_file in tqdm(pred_list, desc="Processing prediction files"):
        pred = standardize_pred_keys(pred_file)
        
        for i, route in enumerate(pred['routes']):
            # Skip if we've processed all target molecules
            if current_idx >= len(target_mols):
                break
                
            current_mol = target_mols[current_idx]
            time = pred['time'][i]
            success = pred['success'][i]


            # Only calculate feasibility for successful routes with valid time
            if success == 1 and time < DEFAULT_TIMEOUT:
                reactions_list = convert_eg_mcts_route(route)
                if reactions_list:
                    try:
                        # Evaluate reaction feasibility
                        reactants_batch, product_batch = featurizer.featurize_reactions_batch(reactions_list)
                        with torch.no_grad():
                            output = rfm_model(reactants=reactants_batch.to(DEVICE),
                                            products=product_batch.to(DEVICE))
                            output = torch.sigmoid(output)
                        
                        # Calculate feasibility scores
                        rfm_score = (output).sum().item()
                        route_feasibility = rfm_score / len(output)
                        
                        # Store results
                        stats['total_reactions'] += len(output)
                        stats['total_rfm_score'][current_idx] = str(output)
                        stats['feasibility'][current_idx] = (output).sum().item() / len(output)

                        
                        # Calculate yields
                        yields = calculate_yields(yield_model, reactions_list)
                        stats['yield_scores'][current_idx] = np.prod(yields)
                        
                        # Success statistics
                        stats['success_rate'][current_idx] = 1
                        stats['depths'][current_idx] = pred['depth'][i]
                        stats['model_calls'][current_idx] = pred['model_calls'][i]
                        stats['search_times'][current_idx] = time
                        
                    except Exception as e:
                        print(f"Error processing molecule {current_idx}: {e}")
            
            current_idx += 1

    # Print results
    print('\n###### VALID RESULTS ######')
    print(f"Total predictions: {total_predictions}")
    
    # Calculate success rate
    print(f"Success rate: {np.mean(stats['success_rate']):.4f}")
    
    # Calculate feasibility scores (only for routes where it was calculated)
    feasibility_mask = np.array(stats['feasibility']) >= 0
    if np.any(feasibility_mask):
        print(f"Feasibility score (route average): {np.mean(np.array(stats['feasibility'])[feasibility_mask]):.4f}")
        print(f"Number of routes with feasibility calculated: {np.sum(feasibility_mask)}")

    # Calculate yield scores (only for routes where it was calculated)
    yield_mask = np.array(stats['yield_scores']) >= 0
    if np.any(yield_mask):
        print(f"Average yield: {np.mean(np.array(stats['yield_scores'])[yield_mask]):.4f}")
    
    # Calculate success statistics (only for successful routes)
    success_mask = np.array(stats['success_rate']) == 1
    if np.any(success_mask):
        print(f"Average depth: {np.mean(np.array(stats['depths'])[success_mask]):.4f}")
        print(f"Average model calls: {np.mean(np.array(stats['model_calls'])[success_mask]):.4f}")
        print(f"Average success inference time: {np.mean(np.array(stats['search_times'])[success_mask]):.4f}")
    
    # Calculate total inference time
    total_search_time = np.mean(stats['search_times'])
    print(f"Average total inference time: {total_search_time:.4f}")
    
    # Save results
    os.makedirs(f'results', exist_ok=True)
    result_path = f'results/{model_name}_{dataset}.json'
    
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f'Results saved to {result_path}')

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main function to parse arguments and run evaluation.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate MCTS predictions')
    parser.add_argument('--tree', '-t', default='MCTS', help='Tree search algorithm')
    parser.add_argument('--ss_model', '-ss', choices=['default', 'AZF', 'LocalRetro', 'Chemformer', 'ReactionT5'])
    parser.add_argument('--dataset', '-d', choices=['n1', 'n5', 'retro190', 'drugbank', 'literature30', 'patent200'])
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(f'results', exist_ok=True)
    # Combine tree and model name
    ss_model = args.ss_model
    dataset = args.dataset
    model_name = f'{args.tree}_{ss_model}'

    # Find and sort prediction files
    pred_files = glob.glob(f'prediction/{model_name}/pred_{dataset}_*.pkl')
    pred_files = sort_pred_files(pred_files)
    
    if not pred_files:
        print(f"No prediction files found for {model_name} on {dataset}")
        
    # Run evaluation
    print(f"\nProcessing {model_name} on {dataset}")
    evaluate_predictions(pred_files, dataset, args, model_name)

if __name__ == '__main__':
    main()