import numpy as np

from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem

def smiles_to_fp(s, fp_dim=4096, pack=False):
    mol = Chem.MolFromSmiles(s)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    fp = morgan_gen.GetFingerprint(mol)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr

def reaction_smarts_to_fp(X, fp_dim=2048):
    rxn = AllChem.ReactionFromSmarts(X)
    # fp = AllChem.CreateDifferenceFingerprintForReaction(rxn)
    fp = AllChem.CreateStructuralFingerprintForReaction(rxn)
    fold_factor = fp.GetNumBits() // fp_dim
    fp = DataStructs.FoldFingerprint(fp, fold_factor)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1
    return arr

def batch_template_to_fp(template_list, fp_dim=2048):
    fps = []
    for t in template_list:
        fps.append(reaction_smarts_to_fp(t, fp_dim))
    fps = np.array(fps)

    assert fps.shape[0] == len(template_list) and fps.shape[1] == fp_dim

    return fps

def batch_smiles_to_fp(s_list, fp_dim=4096):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps

def batch_datas_to_fp(s_list, template_list, fp_dim=2048):
    mol_fps = batch_smiles_to_fp(s_list)
    template_fps = batch_template_to_fp(template_list)
    fps = np.concatenate((mol_fps, template_fps), axis=1)
    # np.array connect
    return fps