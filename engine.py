from torch import nn
import timm
import numpy as np
import torch
from tqdm.notebook import tqdm
from utils import make_loader

def evaluate(data_loader, model):
    
    device        = "cuda:0" if torch.cuda.is_available() else "cpu"   # Setup device
    model.to(device)
    model.eval()
    final_outputs = []
    
    with torch.no_grad():
        
        for (X, y) in tqdm(data_loader, position=0, leave=True, desc='Evaluating'):
            
            X = X.to(device)
            y = y.to(device)
            
            output = model(X)
            output = output.detach().cpu().numpy().tolist()
            
            final_outputs.extend(output)
            
    return final_outputs


def predict(model_dir, config, submission, no_CV_folds):
    
    # Make dataloader for the test set
    test_loader = make_loader(submission, 
                              batch_size = config['batch_size'], 
                              transform  = config['val_transform'], 
                              spatial    = config['spatial_in'],
                              AOnly      = config['AOnly'],
                              balance_sample = False, shuffle = False, num_workers = 4, 
                              pin_memory = True, drop_last = False)

    # Compute predictions of each model
    sig  = nn.Sigmoid()
    outs = []

    for fold in range(no_CV_folds):

        # Load model
        model      = ANN(base_name = config['mdl_base'], pretrained = config['mdl_pretrain'], spatial = config['spatial_in'], AOnly = config['AOnly']);
        checkpoint = torch.load(model_dir + config['exp_name'] + '_fold_' + str(fold) + '.pth')
        model.load_state_dict(checkpoint['model'])

        # Predict
        predictions = evaluate(test_loader, model)
        predictions = np.array(predictions)[:,0]
        
        # Apply sigmoid and append to list
        out = sig(torch.from_numpy(predictions))
        out = out.detach().numpy()
        outs.append(out)
        
    return outs

class EarlyStopping():
    
    def __init__(self, patience = 5, min_delta = 0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        self.save       = True
    
    
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.counter   = 0
            self.best_loss = val_loss
            self.save      = True
            
        elif self.best_loss - val_loss > self.min_delta:
            self.counter   = 0
            self.best_loss = val_loss
            self.save      = True
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.save    = False
            
            if self.counter >= self.patience:
                self.early_stop = True
                self.save    = False

class MixUp(object):
    
    def __init__(self, a = 0.4): # 0.4 -> Suggested in the original paper
        self.a         = a
        self.lamda     = None
        
    def make_multiplier(self):
        
        lamda = np.random.beta(self.a, self.a) # Dims: batch_size x 1
        self.lamda = np.max([lamda, 1 - lamda]) # Avoid duplicates
        
        return
        
    def permute_batch(self, b_size):
        return torch.randperm(b_size)
    
    def compute_target(self, y1, y2):
        # Just for verification
        return y1 * self.lamda + y2 * (1 - self.lamda)
        
    def make_batch(self, X, y):
        
        # Utilising a smart (faster) implementation found here: https://forums.fast.ai/t/mixup-data-augmentation/22764
        idx_arr_s  = self.permute_batch(X.shape[0])

        # Make multiplier for the targets
        self.make_multiplier()

        # Make new predictors
        X = self.lamda * X + (1 - self.lamda) * X[idx_arr_s, :, :, :]

        # Make new targets
        y1, y2 = y, y[idx_arr_s]
        
        # Convert lamda to torch
        return X, y1, y2


# New classifier
class HeadClassifier(nn.Module):
    
    def __init__(self, input_feats):
        super().__init__()
        
        self.fc = nn.Linear(input_feats, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, X):
        
        X = self.fc(X)
        
        return X

# Entire model
class ANN(nn.Module):
    
    def __init__(self, base_name, pretrained = False, spatial = True, AOnly = True):
        super().__init__()
        
        # Get the no. of input channels
        in_channels = ANN.get_input_channels(spatial, AOnly)
        
        # Get baseline
        self.base = timm.create_model(base_name, 
                                      pretrained  = pretrained, 
                                      in_chans    = in_channels, 
                                      num_classes = 1)
        
        try: # EfficientNets
            # Make new classifier
            self.new = HeadClassifier(self.base.classifier.in_features)

            # Forward the input of the baseline classifier
            self.base.classifier = nn.Identity()
            
        except: # NFNet
            # Make new classifier
            self.new = HeadClassifier(self.base.head.fc.in_features)

            # Forward the input of the baseline classifier
            self.base.head.fc = nn.Identity()
            
    
    # Forward phase
    def forward(self, X):
        X = self.base(X)
        X = self.new(X)
        return X
    
    # Freeze all layers of the baseline
    def freeze_base(self):
    
        for _, param in self.base.named_parameters():
            param.requires_grad = False
    
    # Unfreeze all layers
    def unfreeze_all(self):
    
        for _, param in self.named_parameters():
            param.requires_grad = True
    
    @staticmethod
    def get_input_channels(spatial, AOnly):
        
        # Spatial - channel-wise training
        if spatial:
            in_channels = 1
        else:
            if AOnly:
                in_channels = 3
            else:
                in_channels = 6
                
        return in_channels