import os
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree

def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    i = -1
    if not mol_tree.succ:
        for i in range(iterations):
            
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    target_value = m.v_target()
                    scores.append(target_value if target_value is not None else 0.0)  # None일 때는 탐색 가능하도록
                else:
                    scores.append(np.inf)

            scores = np.array(scores)
            metric = scores  # 단순화

            if np.min(metric) == np.inf:
                logging.info('No open nodes!')
                break

            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            result = expand_fn(m_next.mol)

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                
                # if 'templates' in result.keys():
                #     templates = result['templates']
                # else:
                #     templates = result['template']

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs)

                if succ:
                    break

                if (mol_tree.root.succ_value < float('inf') and 
                    mol_tree.search_status < float('inf') and
                    mol_tree.root.succ_value <= mol_tree.search_status):
                    break

            else:
                mol_tree.expand(m_next, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, i+1)


