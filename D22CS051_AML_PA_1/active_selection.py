import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader,Subset

from coreset import Coreset_Greedy

def get_probs(model, loader, device, stochastic=False):
    probs = []
    if stochastic:
        model.train()
    else:
        model.eval()

    count = 0
    with torch.inference_mode():
        for X,y in loader:
            X,y = X.to(device), y.to(device)

            if stochastic:
                output = model.stochastic_pred(X)
            output = model(X)

            # convert log softmax into softmax outputs
            prob = output.cpu().numpy()
            prob = np.exp(prob[0])

            probs.append(prob)

            count += 1

    return np.array(probs)


def get_features(model, loader, device):
    features = []
    model.eval()

    count = 0
    with torch.inference_mode():
        for X,y in loader:
            X, y = X.to(device), y.to(device)
            output = model.get_features(X)
            # pdb.set_trace()
            
            count += 1
            # if count > 10000:
            #     break

            features.append(output.cpu().numpy())
            # features.append((img_name, output.cpu().numpy()))
    return features


def active_sample(
    labeled:list[int],
    unlabeled:list[int],
    sample_size:int,
    dataset: nn.Module,
    method:str='random',
    model:nn.Module=None,
    device:str = 'cpu'
    ):
    '''
    desc: returns 2 list containg labeled samples and unlabled samples lists
    
    '''
    if method == 'random':
        np.random.shuffle(unlabeled)
        sampled = unlabeled[:sample_size]
        labeled.extend(sampled)
        unlabeled = np.setdiff1d(unlabeled,sampled)
        return labeled,unlabeled
    
    if method == 'prob_uncertain' or method == 'prob_margin' or method == 'prob_entropy':
        unlab_loader = DataLoader(Subset(dataset,unlabeled), batch_size=1, shuffle=False) # needed to cal. probabilities.
        probabilities = get_probs(model, unlab_loader, device)
        
        if method == 'prob_uncertain':
            max_probs = np.max(probabilities, axis=1)
            # kind of a heap sort.
            argsorted_maxprobs = np.argpartition(max_probs, sample_size)
            # least probabilities
            sample_indices = argsorted_maxprobs[:sample_size]
            labeled.extend(sample_indices)
            unlabeled = np.setdiff1d(unlabeled,sample_indices)
        
        elif method == 'prob_margin':
            # find the top two probabilities
            top2_sorted = -1 * np.partition(-probabilities, 2, axis=1)
            margins = [x[0]-x[1] for x in top2_sorted]
            margins = np.array(margins)

            # find the ones with highest margin
            argsorted_margins = np.argpartition(-margins, sample_size)
            sample_indices = argsorted_margins[:sample_size]
            labeled.extend(sample_indices)
            unlabeled = np.setdiff1d(unlabeled,sample_indices)

        
        elif method == 'prob_entropy':
            entropy_arr = (-probabilities*np.log2(probabilities)).sum(axis=1)

            # find the ones with the highest entropy
            argsorted_ent = np.argpartition(-entropy_arr, sample_size)
            sample_indices = argsorted_ent[:sample_size]
            labeled.extend(sample_indices)
            unlabeled = np.setdiff1d(unlabeled,sample_indices)
           
        return labeled,unlabeled
    
    if method == 'coreset':
        #create unlabeled loader
        unlab_loader = DataLoader(Subset(dataset,unlabeled), batch_size=1, shuffle=False)

        #labeled dataloader
        lab_loader = DataLoader(Subset(dataset,labeled), batch_size=1, shuffle=False)

        # get labeled features
        labeled_features = get_features(model, lab_loader, device) # (img_name, features)
        # get unlabeled features
        unlabeled_features = get_features(model, unlab_loader, device)# (img_name, features)

        all_features = labeled_features + unlabeled_features
        labeled_indices = np.arange(0,len(labeled_features))

        coreset = Coreset_Greedy(all_features)
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)
        
        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]
        
        sampled = (np.array(unlabeled)[new_batch]).tolist()

        labeled.extend(sampled)
        unlabeled = np.setdiff1d(unlabeled,sampled)
        return labeled,unlabeled
    
    # if method == 'dbal_bald':
    #     # according to BALD implementation by Riashat Islam
    #     # first randomly sample 2000 points
    #     dropout_pool_size = 2000
    #     unl_rows = np.copy(unlabeled_rows)

    #     if len(unl_rows) >= dropout_pool_size:
    #         np.random.shuffle(unl_rows)
    #         dropout_pool = unl_rows[:dropout_pool_size]
    #         temp_unlabeled_csv = 'unlabeled_temp.csv'
    #         np.savetxt(os.path.join(args.dataset_root, temp_unlabeled_csv), dropout_pool,'%s,%s',delimiter=',')
    #         csv_file = temp_unlabeled_csv
    #     else:
    #         dropout_pool = unl_rows
    #         csv_file = 'unlabeled.csv'
        
        

    #     #create unlabeled loader
    #     data_transforms = transforms.Compose([
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.1307,), (0.3081,))
    #                        ])   

    #     unlab_dset = MNIST(args.dataset_root, subset='train',csv_file=csv_file,transform=data_transforms)
    #     unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

    #     scores_sum = np.zeros(shape=(len(dropout_pool), args.nclasses))
    #     entropy_sum = np.zeros(shape=(len(dropout_pool)))

    #     for _ in range(args.dropout_iterations):
    #         probabilities = get_probs(model, unlab_loader, stochastic=True)

            

    #         entropy = - np.multiply(probabilities, np.log(probabilities))
    #         entropy = np.sum(entropy, axis=1)

    #         entropy_sum += entropy
    #         scores_sum += probabilities
            
        
    #     avg_scores = np.divide(scores_sum, args.dropout_iterations)
    #     entropy_avg_sc = - np.multiply(avg_scores, np.log(avg_scores))
    #     entropy_avg_sc = np.sum(entropy_avg_sc, axis=1)

    #     avg_entropy = np.divide(entropy_sum, args.dropout_iterations)

    #     bald_score = entropy_avg_sc - avg_entropy

    #     # partial sort
    #     argsorted_bald = np.argpartition(-bald_score, sample_size)
    #     # get the indices
    #     sample_indices = argsorted_bald[:sample_size]
    #     sample_rows = dropout_pool[sample_indices]

    #     return sample_rows
