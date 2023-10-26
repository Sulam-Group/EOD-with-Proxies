# Necessary functions to run the experiments

# Import necessary packages
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import confusion_matrix

# pytorch
import torch
from torch.optim import Adam
from torch import nn

# BERT from Huggingface
from transformers import BertTokenizer
from transformers import BertModel

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

# Sigmoid function
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

# function to generate ROC curve
def ROC(y,y_prob):
    tprs = []
    fprs = []
    thresholds = np.linspace(0,1,100)
    for t in thresholds:
        y_hat = (y_prob >= t).astype('float')
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(tn+fp))
    return (thresholds, fprs, tprs)

# New function to generate data (X,y)
def generate_data(n,e0,e1,b,t):

    # Mean and variance for features x1,x2,x3
    mu = np.array([1,-1,0])
    var = np.array([[1,0.05,0],[0.05,1,0],[0,0,0.05]])
    X = np.random.multivariate_normal(mu,var,n)

    # Function from x3 to A
    a = ((X[:,2] + b) >= 0).astype('float')

    # Function from x1 and x2 to A
    eps_0 = np.random.normal(0,e0,n)
    eps_1 = np.random.normal(0,e1,n)

    # add noise to a = 0 or a = 1
    noise_a0 = eps_0*(a==0)
    noise_a1 = eps_1*(a==1)

    # Generate y depending on experiment
    p = sigmoid(X[:,0] + X[:,1] + X[:,2] + noise_a0 + noise_a1)
    y = np.random.binomial(n = 1, p = p)
    
    return (X, a, y)

# Function to generate y_prob using random coefficients
def generate_y_hat(X,coeffs,t):
    y_prob = sigmoid(np.dot(X,coeffs))
    y_hat = (y_prob >= t).astype('float')

    return (y_prob, y_hat)

# Generate a_hat
def generate_a_hat(x3, b, e, imbalance = False):
    if imbalance == True:
        noise = e
    else:
        noise = np.random.normal(0,e,len(x3))
    a_hat = ((x3 + b + noise) >= 0).astype('float')
    return a_hat

# Dataset class for BERT
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label for label in df['a']]
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 5, truncation=True,
                                return_tensors="pt") for text in df['long_name']]
        self.remain_data = [df[['age','overall','y','group']].iloc[idx] for idx in range(df.shape[0])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        batch_texts = self.texts[idx]
        batch_y = torch.tensor(self.labels[idx])
        batch_rest = torch.tensor(self.remain_data[idx])

        return batch_texts, batch_y, batch_rest

# Class for classifier
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigmoid(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    
    batch_sz = 2

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_sz, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_sz)

    # os.environ['CUDE_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "mps")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)


    for epoch_num in range(epochs):

        model.train()
        total_loss_train = 0
        total_tp_train = 0

        for train_input, train_label, _ in tqdm(train_dataloader):

            train_label = train_label.to(device).float()
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask).reshape(1,-1)[0] 
            batch_loss = criterion(output, train_label)
            batch_tp = torch.sum((output >= 0.5) == train_label)
            
            total_tp_train += batch_tp.item()
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
                
        total_loss_val = 0
        total_tp_val = 0

        with torch.no_grad():

            model.eval()

            for val_input, val_label, _ in val_dataloader:

                val_label = val_label.to(device).float()
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask).reshape(1,-1)[0]
                batch_loss = criterion(output, val_label)
                batch_tp = torch.sum((output >= 0.5) == val_label)
                    
                total_tp_val += batch_tp.item()    
                total_loss_val += batch_loss.item()
                    
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_data)/batch_sz): .3f} \
            | Train Accuracy: {total_tp_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / (len(val_data)/batch_sz): .3f} \
            | Val Accuracy: {total_tp_val / len(val_data): .3f}')