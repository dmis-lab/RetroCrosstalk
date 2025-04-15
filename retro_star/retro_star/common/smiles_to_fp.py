import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    fp = morgan_gen.GetFingerprint(mol)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps