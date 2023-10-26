import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from copy import deepcopy

class CLFRates:
    def __init__(self,
                 y, 
                 y_,
                 round=4):
        self.tab = confusion_matrix(y, y_)
        tn = self.tab[0, 0]
        fn = self.tab[1, 0]
        fp = self.tab[0, 1]
        tp = self.tab[1, 1]
        self.num_pos = (tp + fn)/len(y)
        self.num_neg = (fp + tn)/len(y)
        self.pr = (tp + fp)/len(y)
        self.nr = (tn + fn)/len(y)
        self.tnr = tn/(tn + fp)
        self.tpr = tp/(tp + fn)
        self.fnr = fn /(fn + tp)
        self.fpr = fp/(fp + tn)
        self.acc = (tn + tp)/len(y)

def pred_from_pya(y_, a, pya, binom=False):
    # Getting the groups and making the initially all-zero predictor
    groups = np.unique(a)
    out = deepcopy(y_)
    
    for i, g in enumerate(groups):
        group_ids = np.where((a == g))[0]
        
        # Pulling the fitted switch probabilities for the group
        p = pya[i]
        
        # Indices in the group from which to choose swaps
        pos = np.where((a == g) & (y_ == 1))[0]
        neg = np.where((a == g) & (y_ == 0))[0]
        
        if not binom:
            # Randomly picking the positive predictions
            pos_samp = np.random.choice(a=pos, 
                                        size=int(p[1] * len(pos)), 
                                        replace=False)
            neg_samp = np.random.choice(a=neg, 
                                        size=int(p[0] * len(neg)),
                                        replace=False)
            samp = np.concatenate((pos_samp, neg_samp)).flatten()
            out[samp] = 1
            out[np.setdiff1d(group_ids, samp)] = 0
    
    return out.astype(np.uint8)

'''
# Quick function for thresholding probabilities
def threshold(probs, cutoff=.5):
    return np.array(probs >= cutoff).astype(np.uint8)

    def loss_from_roc(y, probs, roc):
    points = [(roc[0][i], roc[1][i]) for i in range(len(roc[0]))]
    guess_list = [threshold(probs, t) for t in roc[2]]
    accs = [accuracy_score(y, g) for g in guess_list]
    js = [p[1] - p[0] for p in points]
    tops = [from_top(point) for point in points]
    return {'guesses': guess_list, 
            'accs': accs, 
            'js': js, 
            'tops': tops}

def from_top(roc_point, round=4):
    d = np.sqrt(roc_point[0]**2 + (roc_point[1] - 1)**2)
    return d

def roc_coords(y, y_, round=4):

    # Getting the counts
    tab = confusion_matrix(y, y_)
    tn = tab[0, 0]
    fn = tab[1, 0]
    fp = tab[0, 1]
    tp = tab[1, 1]
    # Calculating the rates
    tpr = np.round(tp / (tp + fn), round)
    fpr = np.round(fp / (fp + tn), round)
    
    return (fpr, tpr)

def group_roc_coords(y, y_, a, round=4):
    groups = np.unique(a)
    group_ids = [np.where(a ==g)[0] for g in groups]
    coords = [roc_coords(y[i], y_[i], round) for i in group_ids]
    fprs = [c[0] for c in coords]
    tprs = [c[1] for c in coords]
    out = pd.DataFrame([groups, fprs, tprs]).transpose()
    out.columns = ['group', 'fpr', 'tpr']
    return out
'''