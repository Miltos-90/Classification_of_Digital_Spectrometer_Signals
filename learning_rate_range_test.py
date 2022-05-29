
from engine import MixUp
from engine import ANN
from utils import make_loader
from tqdm.notebook import tqdm
import pandas as pd
from torch import nn, optim
from torch.cuda import amp
import torch
import matplotlib.pyplot as plt
import itertools


# Small wrapper to run multiple LR tests
def run_LR_tests(df, skf, config, batch_sizes, w_decays, fold_no = 0):
    
    LRT, res = LRTest(df, skf), []
    for (batch_size, w_decay) in itertools.product(batch_sizes, w_decays):
        
        config['batch_size'], config['weight_decay'] = batch_size, w_decay
        res.append(LRT(config, fold = fold_no))

    return pd.concat(res, axis = 1) # Make dataframe from experiments


class LRTest(object):
    
    def __init__(self, data, skf, min_lr = 1e-6, max_lr = 1, no_iter = 100):
        self.no_iter    = no_iter
        self.max_lr     = max_lr
        self.min_lr     = min_lr
        self.device     = None
        self.mixup      = None
        self.data       = data
        self.fold_ids   = skf
        self.mult       = (max_lr / min_lr) ** (1 / (no_iter - 1))
        self.dataiter   = None
    
    
    def init_train(self, config, fold):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"              # Setup device
        self.mixup  = MixUp(config['mixup_A']) if config['mixup_A'] is not None else None # Setup mixup
        criterion   = nn.BCEWithLogitsLoss()                                        # Criterion
        scaler      = amp.GradScaler() if config['amp_enable'] else None            # Scaler
        model       = ANN(base_name  = config['mdl_base'],                          # Model
                          pretrained = config['mdl_pretrain'], 
                          spatial    = config['spatial_in'], 
                          AOnly      = config['AOnly']);
        if config['mdl_freeze']: 
            model.freeze_base()
            
        optimizer = optim.AdamW(model.parameters(), lr = self.min_lr, weight_decay = config['weight_decay'])
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda x: self.mult)

        # Make dataloader
        train_idx, val_idx = self.fold_ids[fold] # Grab data for this fold
        dataloader = make_loader(self.data.loc[train_idx], 
                                   batch_size = config['batch_size'],
                                   transform  = config['train_transform'],
                                   spatial    = config['spatial_in'],
                                   AOnly      = config['AOnly'],
                                   balance_sample = False, shuffle = True, num_workers = 4, pin_memory = True, drop_last = True)
        self.dataiter = iter(dataloader)
        
        return model, criterion, scaler, optimizer, scheduler, dataloader
    
    
    # Function to perform the learning rate range test on one experiment
    def __call__(self, config, fold):
        
        model, criterion, scaler, optimizer, scheduler, dataloader = self.init_train(config, fold)
        loss_arr, lr_arr     = [], [] 
        cur_iter, best_loss  = 0, 1e9
        model.to(self.device);

        with tqdm(total = self.no_iter) as pbar:
            
            while cur_iter < self.no_iter:
                
                # Grab last learning rate (before stepping the scheduler)
                lr_arr.append(scheduler.get_last_lr()[0])
                
                # Train a batch
                cur_loss = self.train_batch(model, criterion, optimizer, scheduler, scaler, dataloader)
                
                # Append loss/learning rate to arrays
                loss_arr.append(cur_loss)

                # Check for divergence and exit if needed
                if cur_loss < best_loss: 
                    best_loss = cur_loss

                if cur_loss > 2e2 * best_loss: # Divergence
                    print('Diverged on iteration ' + str(cur_iter) + ' with loss ' + str(cur_loss))
                    break

                # Update progress bar
                pbar.set_postfix(loss = cur_loss)
                pbar.update(1)
                cur_iter += 1

        pbar.close() # Close

        # Make dataframe with results
        res = pd.DataFrame({"lr" : lr_arr, "train_loss" : loss_arr}).set_index('lr')
        res.columns = ['BTS = ' + str(config['batch_size']) + ', WD = ' + str(config['weight_decay'])]

        return res
    
    
    # Return a batch
    def grab_batch(self, dataloader):
            
        try:
            X, y = next(self.dataiter)
        except StopIteration: # End of dataset -> restart
            self.dataiter = iter(dataloader)
            X, y     = next(self.dataiter)
                
        return X, y
    
    
    # Train batch
    def train_batch(self, model, criterion, optimizer, scheduler, scaler, dataloader):

        model.train()
        optimizer.zero_grad()
        cur_loss = 0
            
        if self.mixup is not None:
            cur_loss += self.train_mixup(model, criterion, scaler, dataloader)    
        else:
            cur_loss += self.train_plain(model, criterion, scaler, dataloader)
                    
        # Update all
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        cur_loss = cur_loss.item()

        return cur_loss
    
    
    # Train a batch w/ MixUp loss
    def train_mixup(self, model, criterion, scaler, dataloader):
            
        X, y      = self.grab_batch(dataloader)
        X, y1, y2 = self.mixup.make_batch(X, y)
        X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)
                    
        if scaler is not None:
            with amp.autocast():
                y_hat = model(X)
                loss  = self.mixup.lamda * criterion(y_hat, y1) + (1 - self.mixup.lamda) * criterion(y_hat, y2)
                scaler.scale(loss).backward()
        else:
            y_hat = model(X)
            loss  = self.mixup.lamda * criterion(y_hat, y1) + (1 - self.mixup.lamda) * criterion(y_hat, y2)
            loss.backward()
                
        return loss
    
    
    # Train a batch w/ actual loss
    def train_plain(self, model, criterion, scaler, dataloader):
            
        X, y = self.grab_batch(dataloader)
        X, y = X.to(self.device), y.to(self.device)
                    
        if scaler is not None:
            with amp.autocast():
                y_hat = model(X)
                loss  = criterion(y_hat, y)
                scaler.scale(loss).backward()
        else:
            y_hat = model(X)
            loss  = criterion(y_hat, y)
            loss.backward()
                
        return loss
    
    
    @staticmethod
    def plot_LR_test(results_df, rolling_window, figsize):
        
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        results_df.rolling(window = rolling_window).mean().plot(kind = 'line', marker = None, logx = True, legend = True, grid = True, ax = ax);
        ax.set_title('Learning Rate Range Test')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate')
        
        return