from typing import List, Tuple

import dgl
from dgl import DGLGraph
from dgllife.utils import CanonicalBondFeaturizer, WeaveAtomFeaturizer, mol_to_bigraph
from .utils import ATOM_TYPES
from rdkit import Chem


class ReactionFeaturizer:
    def __init__(self):
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=ATOM_TYPES)

    def featurize_smiles_single(self, smiles: str) -> DGLGraph:
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveHs(mol)
        return mol_to_bigraph(
            mol=mol,
            add_self_loop=True,
            canonical_atom_order=False,
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
        )

    def featurize_reaction_single(
        self, reactants: List[str], product: str
    ) -> Tuple[DGLGraph, DGLGraph]:
        # try:
        reactants_graphs = dgl.merge([self.featurize_smiles_single(r) for r in reactants])
        # except:
        #     import pdb; pdb.set_trace()
        product_graph = self.featurize_smiles_single(product)
        return reactants_graphs, product_graph

    def featurize_reactions_batch(
        self, reaction_list: List[Tuple[List[str], str]]
    ) -> Tuple[DGLGraph, DGLGraph]:
        reactions = []
        for reactants, product in reaction_list:
            try:
                reaction = self.featurize_reaction_single(reactants, product)
                reactions.append(reaction)
            except:
                pass
        return self.collate_reactions(reactions)

    def collate_reactions(
        self, reactions: List[Tuple[DGLGraph, DGLGraph]]
    ) -> Tuple[DGLGraph, DGLGraph]:
        # try:
        reactants, products = zip(*reactions)
        # except:
        #     import pdb; pdb.set_trace()
        reactants_batch = dgl.batch(reactants)
        product_batch = dgl.batch(products)
        return reactants_batch, product_batch
