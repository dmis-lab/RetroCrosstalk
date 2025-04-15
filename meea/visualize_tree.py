"""
Retrosynthesis Reaction Tree Parser and Visualizer

This script processes retrosynthesis reaction pathways, converts them into tree structures,
and visualizes the resulting synthetic trees. It supports different reaction formats and
can process various datasets of chemical reactions.
"""

import os
import re
import glob
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from visualize import draw_tree_from_path_string


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
        """Recursively build retrosynthesis tree"""
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
    """Format tree for readable display"""
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


def test_parser(reactions_list):
    """Debug function to test the parser"""
    print("Reaction sequence:")
    for i, reaction in enumerate(reactions_list):
        reactants, products = reaction.strip().split('>>')
        print(f"Step {i+1}:")
        print(f"  Reactants: {reactants}")
        print(f"  Products: {products}")
        print()
    
    tree = parse_reactions(reactions_list)
    print("Generated tree:")
    print(format_tree(tree))
    return tree


def process_reactions(reactions_list, mode='rp'):
    """Process reaction list and return formatted retrosynthesis tree"""
    tree = parse_reactions(reactions_list, mode)
    formatted_tree = format_tree(tree)
    return formatted_tree


def sort_pred_files(files):
    """Sort prediction files by numerical index"""
    def extract_numbers(filename: str) -> tuple:
        numbers = re.findall(r'(\d+)_(\d+)', filename)
        if numbers:
            return tuple(map(int, numbers[0]))
        return (0, 0) 
    
    sorted_files = sorted(files, key=extract_numbers)
    return sorted_files


def format_reaction_path(data, paths=None, current_path=None, in_stock_list=None):
    """
    Format reaction path from tree data
    
    Args:
        data: Tree data dictionary
        paths: List to collect reaction paths
        current_path: Current path being processed
        in_stock_list: List of in-stock molecules
    
    Returns:
        Tuple of (paths, in_stock_list)
    """
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []
    if in_stock_list is None:
        in_stock_list = []
    
    if not isinstance(data, dict):
        return paths
    
    # Store molecules marked as in_stock
    if 'smiles' in data and 'in_stock' in data:
        if data['in_stock'] is True and data['smiles']:
            if data['smiles'] not in in_stock_list:
                in_stock_list.append(data['smiles'])
    
    # Add current node's SMILES to path
    if 'smiles' in data and data['smiles']:
        current_path.append(data['smiles'])
    
    # Generate reaction path for nodes with children
    if 'children' in data and data['children']:
        reactants = []
        for child in data['children']:
            if 'smiles' in child and child['smiles']:
                reactants.append(child['smiles'])
            # Recursively explore sub-paths
            format_reaction_path(child, paths, current_path[:], in_stock_list)
        
        # Add reaction if reactants exist and current path has a product
        if reactants and current_path:
            reaction = f"{'.'.join(reactants)} >> {current_path[-1]}"
            paths.append(reaction)
    
    return paths, in_stock_list


if __name__ == "__main__":
    # Configuration
    folder = 'meea'
    algorithm = 'MEEA'
    ss_model = 'default'
    dataset = 'n1'
    mode = 'rp'
    
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
    
    # Load gold standard data based on dataset
    if 'n1' in dataset:
        gold = json.load(open('../Data/PaRoutes/n1-routes.json', 'r'))
        gold = [format_reaction_path(gold_path)[0] for gold_path in gold]
    elif 'n5' in dataset:
        gold = json.load(open('../Data/PaRoutes/n5-routes.json', 'r'))
        gold = [format_reaction_path(gold_path)[0] for gold_path in gold]
    elif 'literature' in dataset:
        def literature_sort_pred_files(files):
            def extract_numbers(filename: str) -> tuple:
                numbers = re.findall(r'(\d+)_liter', filename)
                if numbers:
                    return tuple(map(int, numbers[0]))
                return 0 
            
            sorted_files = sorted(files, key=extract_numbers)
            return sorted_files
        
        gold_list = glob.glob('../Data/Literature30/*_liter.txt')
        gold_list = literature_sort_pred_files(gold_list)
        gold = []
        for g in gold_list:
            with open(g, 'r') as f:
                gold.append(f.readlines())
    elif 'retro' in dataset:
        gold = pickle.load(open('../Data/Retro190/retro190_routes.pkl', 'rb'))
    

    print(f'{algorithm}_{ss_model}_{dataset}\n')
    mol_num = 0
    pred_list = glob.glob(f'./prediction/{algorithm}_{ss_model}/pred_{dataset}*.pkl')
    pred_list = sort_pred_files(pred_list)
    fig_path = f'figures/{algorithm}_{ss_model}'
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(f'{fig_path}/{dataset}', exist_ok=True)
    
    # Count total molecules
    total_mols = 0
    for pred in pred_list:
        pred = pickle.load(open(pred, 'rb'))
        total_mols += len(pred['success'])
    
    # Validate total count
    print(f"Expected {num_mols[dataset]} molecules, found {total_mols}")
    print()
    #assert total_mols == num_mols[dataset]

    
    # Process prediction files
    for pred in tqdm(pred_list):
        pred = pickle.load(open(pred, 'rb'))
        routes = pred['routes']
        
        # Process individual routes
        for i, route in enumerate(routes):
            mol_num += 1
            
            # Skip invalid routes
            if route is None or len(route) <= 1:
                continue
            
            # Process route (skip the first element)
            route = route[1:]
            
            try:
                # Parse reactions into tree structure
                tree_dict = process_reactions(route, mode=mode)
            except Exception as e:
                print(f'{mol_num} parse failed: {str(e)}')
                print(route)
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