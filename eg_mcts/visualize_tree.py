"""
MCTS (Monte Carlo Tree Search) Reaction Pathway Visualization

This script processes MCTS retrosynthesis route predictions, converts them into tree structures,
and generates visualizations of the synthetic pathways. It supports multiple models and datasets.
"""

import os
import re
import glob
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from visualize import draw_tree_from_path_string
from alg.syn_route import SynRoute

# =============================================================================
# Reaction Parsing Functions
# =============================================================================

def split_in_place(s, delimiter1=".", delimiter2=">>"):
    """
    Split reaction SMILES into reactants and product.
    
    Args:
        s: Reaction SMILES string
        delimiter1: Delimiter between reactants (default: ".")
        delimiter2: Delimiter between reactants and product (default: ">>")
        
    Returns:
        Tuple of (reactants_list, product)
    """
    if delimiter2 in s:
        b, c = s.split(delimiter2)
        if delimiter1 in b:
            a, b = b.split(delimiter1, 1)
            reactants = [a, b]
        else:
            reactants = [b]
        return (reactants, c)
    return None

def convert_eg_mcts_route(routes):
    """
    Convert MCTS route object to a list of reaction SMILES strings.
    
    Args:
        routes: Route object from MCTS algorithm
        
    Returns:
        List of reaction SMILES strings or None if routes is None
    """
    reactions_list = None
    if routes is not None: 
        route_paths = []
        for i in range(routes.length):
            reaction = SynRoute.serialize_reaction(routes, i)
            if '>>' in reaction:
                product, reactant = reaction.split('>>')
                reaction = reactant + '>>' + product
                route_paths.append(reaction)
        reactions_list = route_paths
    return reactions_list
    
def parse_reactions(reactions_list, mode='rp'):
    """
    Convert a list of retrosynthesis reactions into a tree structure.
    
    Args:
        reactions_list: List of reaction SMILES strings
        mode: Reaction direction - 'rp' (reactants>>product) or 'pr' (product>>reactants)
    
    Returns:
        Dictionary representation of the retrosynthesis tree
    """
    if None in reactions_list:
        reactions_list = [r for r in reactions_list if r is not None]
        
    # Create reaction graph
    graph = {}  # product -> reactants mapping
    all_products = set()
    all_reactants = set()
    
    # Build reaction graph
    for reaction in reactions_list:
        if mode == 'pr':
            product, reactants = reaction.split('>>')
        else:  # mode == 'rp'
            reactants, product = reaction.split('>>')
            
        product = product.strip()
        reactants = [r.strip() for r in reactants.split('.')]
        
        # Add to reaction graph
        if product not in graph:
            graph[product] = []
        graph[product].append(reactants)
        
        # Collect all products and reactants
        all_products.add(product)
        all_reactants.update(reactants)

    # Find target products - products that are not used as reactants
    target_products = all_products - all_reactants

    def build_tree(product, visited=None):
        """
        Recursively build retrosynthesis tree
        
        Args:
            product: Current product SMILES
            visited: Set of already visited molecules to prevent cycles
            
        Returns:
            Tree dictionary for the current product
        """
        if visited is None:
            visited = set()
            
        if product in visited:  # Prevent cycles
            return {'smiles': product}
            
        current_visited = visited | {product}
        
        if product not in graph:  # Terminal reactant that can't be broken down further
            return {'smiles': product}
            
        children = []
        # For all possible reactant sets for this product
        for reactants in graph[product]:
            for reactant in reactants:
                if reactant not in visited:
                    child_tree = build_tree(reactant, current_visited)
                    if child_tree:
                        children.append(child_tree)
                        
        if children:
            return {
                'smiles': product,
                'children': children
            }
        return {'smiles': product}

    # Backup strategy for cases without target products
    if not target_products:
        # Choose the most complex (longest) SMILES string as target
        target = max(all_products, key=len)
        tree = build_tree(target)
        return tree if 'children' in tree else {}

    # Select target product with longest path
    best_tree = None
    max_depth = 0
    
    def get_tree_depth(tree):
        """
        Calculate the depth of a tree
        
        Args:
            tree: Tree dictionary
            
        Returns:
            Depth as an integer
        """
        if 'children' not in tree:
            return 1
        return 1 + max(get_tree_depth(child) for child in tree['children'])

    for target in target_products:
        tree = build_tree(target)
        if 'children' in tree:  # Only consider trees with children
            depth = get_tree_depth(tree)
            if depth > max_depth:
                max_depth = depth
                best_tree = tree

    return best_tree if best_tree else build_tree(max(all_products, key=len))

def format_tree(tree, indent=0):
    """
    Format tree for readable display
    
    Args:
        tree: Tree dictionary
        indent: Current indentation level
        
    Returns:
        Formatted string representation of the tree
    """
    if not tree:
        return "{}"
        
    result = "{"
    result += f"'smiles': '{tree['smiles']}'"
    
    if 'children' in tree and tree['children']:
        result += ", 'children': ["
        for i, child in enumerate(tree['children']):
            result += "\n" + "  " * (indent + 1) + format_tree(child, indent + 1)
            if i < len(tree['children']) - 1:
                result += ","
        result += "\n" + "  " * indent + "]"
    
    result += "}"
    return result

def process_reactions(reactions_list, mode='rp'):
    """
    Process reaction list and return formatted retrosynthesis tree
    
    Args:
        reactions_list: List of reaction SMILES strings
        mode: Reaction direction ('rp' or 'pr')
        
    Returns:
        Formatted tree string
    """
    tree = parse_reactions(reactions_list, mode)
    formatted_tree = format_tree(tree)
    return formatted_tree

def sort_pred_files(files):
    """
    Sort prediction files by numerical index
    
    Args:
        files: List of filenames to sort
        
    Returns:
        Sorted list of filenames
    """
    def extract_numbers(filename: str) -> tuple:
        numbers = re.findall(r'(\d+)_(\d+)', filename)
        if numbers:
            return tuple(map(int, numbers[0]))
        return (0, 0) 
    
    sorted_files = sorted(files, key=extract_numbers)
    return sorted_files

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main function to process and visualize MCTS reaction pathways
    """
    # Configuration
    folder = 'eg_mcts'
    algorithm = 'MCTS'
    mode = 'rp'  # reactants>>product mode
    ss_model = 'default'
    dataset = 'literature30'
    # Dataset sizes
    num_mols = {
        'n1': 10000,
        'n5': 10000,
        'retro190': 190,
        'drugbank': 8313,
        'literature30': 30,
        'patent200': 200,
    }


    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Process each model and dataset
    print(f'{algorithm}_{ss_model}_{dataset}\n')
    
    # Initialize counter and find prediction files
    mol_num = 0
    pred_list = glob.glob(f'./prediction/{algorithm}_{ss_model}/pred_{dataset}*.pkl')
    pred_list = sort_pred_files(pred_list)
    
    # Create output directories
    fig_path = f'figures/{algorithm}_{ss_model}'
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(f'{fig_path}/{dataset}', exist_ok=True)

    # Count total molecules for validation
    total_mols = 0
    for pred in pred_list:
        with open(pred, 'rb') as f:
            pred_data = pickle.load(f)
        total_mols += len(pred_data['success'])
    
    # Validate molecule count
    print(f"Expected {num_mols[dataset]} molecules, found {total_mols}")
    print()
    assert total_mols == num_mols[dataset]
        
    # Process each prediction file
    for pred_file in tqdm(pred_list, desc=f"Processing {ss_model} files"):
        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)
        
        # Extract routes using standardized key names
        routes = pred['routes']

        
        # Process each route
        for i, route in enumerate(routes):
            mol_num += 1
            
            # Skip invalid routes
            if route is None:
                continue
            
            # Convert route format
            route_paths = convert_eg_mcts_route(route)
            if not route_paths:
                continue
            
            try:
                # Parse reactions into tree structure
                tree_dict = process_reactions(route_paths, mode=mode)
            except Exception as e:
                print(f'{mol_num} parse failed: {str(e)}')
                print(route_paths)
                print()
                continue
                
            try:
                # Generate and save tree visualization
                output_path = Path(f"{fig_path}/{dataset}/{dataset}_{mol_num-1}_tree")
                draw_tree_from_path_string(
                    path_string=str(tree_dict),
                    save_path=output_path,
                    y_margin=150
                )
            except Exception as e:
                print(f'{mol_num} figure generation failed: {str(e)}')
                print(tree_dict)
                print()

if __name__ == "__main__":
    main()