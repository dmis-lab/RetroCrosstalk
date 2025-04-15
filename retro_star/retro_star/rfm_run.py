import torch
import pickle
import glob
import numpy as np
import re
from tqdm import tqdm
from ReactionFeasibilityModel.rfm.featurizers.featurizer import ReactionFeaturizer
from ReactionFeasibilityModel.rfm.models import ReactionGNN

def split_in_place_meea(s, delimiter1, delimiter2):
    # Split once on the first delimiter to get a and b
    if delimiter1 in s:
        a, b = s.split(delimiter1, 1)
        b, c = b.split(delimiter2, 1)
        reactants = [a,b]
    else:
        a,c = s.split(delimiter2, 1)
        reactants = [a]
    return (reactants, c)


def split_in_place_egmcts(s, delimiter1, delimiter2):
    # Split once on the first delimiter to get a and b
    if delimiter1 in s:
        a, b = s.split(delimiter1, 1)
        c, a = a.split(delimiter2, 1)
        reactants = [a,b]
    else:
        c, a = s.split(delimiter2, 1)
        reactants = [a]
    return (reactants, c)


#files = ['/hdd1/yan/MEEA_model/MEEA_model/test/stat_pc_retro190_(targets,routes)_0_190_4.0.pkl']
# files = ['results/n1_0_3.pkl']
# print(files)



def preprocess_meea_route(route):
    if len(route) >1:
        reactions_list = [split_in_place_meea(i.strip(), '.', '>>') for i in route]
    return reactions_list

def preprocess_egmcts_route(route):
    try:
        route = route.serialize().split('|')
        reactions_list = [split_in_place_egmcts(i.strip(), '.', '>>') for i in route]
    except:
        reactions_list = []
    return reactions_list

def preprocess_retrostar_route(route):
    try:
        route = route.split('|')
        route = [re.sub(f'>[^>]*>', '>>', i) for i in route]
        reactions_list = [split_in_place_egmcts(i.strip(), '.', '>>') for i in route]
    except:
        reactions_list = []
    return reactions_list


checkpoint = 'ReactionFeasibilityModel/checkpoints/eval/best_reaction.pt'
device = torch.device('cuda')

def get_stat(pred_list):
    model = ReactionGNN(checkpoint_path=checkpoint)
    featurizer = ReactionFeaturizer()
    model.eval()
    model = model.to(device)


    success_rate = []
    depths = []
    model_calls = []
    feasibility_score = []
    rfm_scores = []
    total_route = 0

    for p in tqdm(pred_list):
        try:
            pred = pickle.load(open(p, 'rb'))
        except:
            print(p)
            continue
        # success_rate += pred['success']
        # depths += [d for d in pred['depth'] if d!=32]
        # model_calls += [d for d in pred['counts'] if d !=-1]
        success_rate += pred['succ']
        depths += [d for d in pred['route_len'] if d not in [None, 32]]
        model_calls += [d for d in pred['iter'] if d not in [None, 200]]

        for route in tqdm(pred['routes']):
            reactions_list = preprocess_retrostar_route(route)
            if len(reactions_list) > 0:
                reactants_batch, product_batch = featurizer.featurize_reactions_batch(reactions_list)
                reactants_batch = reactants_batch.to(device)
                product_batch = product_batch.to(device)
                with torch.no_grad():
                    output = model.forward(reactants=reactants_batch, products=product_batch)
                rfm_score = (output > 0.9).sum().item()
                rfm_scores.append(rfm_score)
                total_route += len(output)
                feasibility_score.append(rfm_score/len(output))
    print(len(success_rate))    
    print(np.mean(success_rate))
    print(np.mean(depths))
    print(np.mean(model_calls))

    print(f'Feasibiltiy Score averaged by route number : {np.mean(feasibility_score) , len(feasibility_score)}')
    print(f'Feasibiltiy Score averaged by reaction number : {np.sum(rfm_scores) / total_route, total_route}') 

    

model_name = 'RETRO_ReactionT5'
dataset = 'retro190'
#pred_list = glob.glob(f'prediction/{model_name}/pred_pc_{dataset}*')
pred_list = glob.glob(f'../prediction/{model_name}/*pred_{dataset}_*')
print(pred_list)
get_stat(pred_list)




