from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import torch
import numpy as np
import math



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

def value_fn_default(model, mols, device):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
    index = len(fps)
    if len(fps) <= 5:
        mask = np.ones(5)
        mask[index:] = 0
        fps_input = np.zeros((5, 2048))
        fps_input[:index, :] = fps
    else:
        mask = np.ones(len(fps))
        fps_input = fps
    fps = torch.from_numpy(np.array(fps_input, dtype=np.float32)).unsqueeze(0).to(device)
    mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0).to(device)
    v = model(fps, mask).cpu().data.numpy()
    return v[0][0]
