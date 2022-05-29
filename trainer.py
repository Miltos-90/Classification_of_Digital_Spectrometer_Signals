from utils import make_loader
from engine import ANN, MixUp, EarlyStopping
from torch import optim
import torch
from torch import nn
from torch.cuda import amp
import numpy as np
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

# Train / validate
class Trainer():
    
    
    def __init__(self, data, fold_ids, save = False):
        self.data              = data
        self.skf               = fold_ids
        self.save              = save
        self.chekpoint_dirname = None
        self.plot_dirname      = None
        self.pred_dirname      = None
    
    
    # Make dataloaders
    def make_dataloaders(self, config, fold):
        
        train_idx, val_idx = self.skf[fold] # Grab data for this fold
        
        train_loader = make_loader(self.data.loc[train_idx], 
                                   batch_size = config['batch_size'], 
                                   transform  = config['train_transform'], 
                                   spatial    = config['spatial_in'], 
                                   AOnly      = config['AOnly'], 
                                   balance_sample = False, shuffle = True, num_workers = 4, pin_memory = True, drop_last = True)

        val_loader   = make_loader(self.data.loc[val_idx], 
                                   batch_size = config['batch_size'], 
                                   transform  = config['val_transform'], 
                                   spatial    = config['spatial_in'], 
                                   AOnly      = config['AOnly'], 
                                   balance_sample = False, shuffle = False, num_workers = 4, pin_memory = True, drop_last = False)
        
        return train_loader, val_loader
    
    
    # Setup model
    @staticmethod
    def make_model(config):
        
        model = ANN(base_name = config['mdl_base'], pretrained = config['mdl_pretrain'], spatial = config['spatial_in'], AOnly = config['AOnly']);
        
        if config['mdl_freeze']: model.freeze_base()
            
        return model
    
    
    # Setup the optimiser
    @staticmethod
    def make_optimizer(model, learn_rate, weight_decay):
        
        optimizer = optim.AdamW(model.parameters(), lr = learn_rate, weight_decay = weight_decay)
        
        return optimizer

    @staticmethod
    def make_scheduler(optimizer, train_loader, config):
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                  max_lr          = config['learn_rate'], 
                                                  steps_per_epoch = len(train_loader), epochs = config['no_epochs'], 
                                                  anneal_strategy = 'cos', 
                                                  cycle_momentum  = False, 
                                                  pct_start       = config['pct_start'], 
                                                  div_factor      = config['div_factor'])
        
        return scheduler
    
    
    # Setup training
    def setup_train(self, config, fold):

        # Make directories
        self.chekpoint_dirname = '/kaggle/working/' + config['exp_name'] + '_fold_' + str(fold) + '.pth' # Directory to save the model state dict etc.
        self.plot_dirname      = '/kaggle/working/' + config['exp_name'] + '_fold_' + str(fold) + '.png' # Directory to save plot
        self.pred_dirname      = '/kaggle/working/' + config['exp_name'] + '_fold_' + str(fold) + '.npy' # Directory to save plot
        
        # Seup objects
        device     = "cuda:0" if torch.cuda.is_available() else "cpu"   # Setup device
        criterion  = nn.BCEWithLogitsLoss()                             # Criterion
        scaler     = amp.GradScaler() if config['amp_enable'] else None # Scaler
        model      = self.make_model(config)
        mixup      = MixUp(config['mixup_A']) if config['mixup_A'] is not None else None 
        early_stop = EarlyStopping(config['patience'], config['min_delta']) if config['early_stop'] else None
        train_loader, val_loader = self.make_dataloaders(config, fold) # Data loaders

        # Setup scheduler and optimizer
        if config['scheduler'] == 'OneCycle':
            optimizer = self.make_optimizer(model, config['learn_rate'] / config['div_factor'], config['weight_decay'])
            scheduler = self.make_scheduler(optimizer, train_loader, config)
        else:
            optimizer = self.make_optimizer(model, config['learn_rate'], config['weight_decay'])
            scheduler = None

        return device, mixup, model, criterion, scaler, optimizer, train_loader, val_loader, scheduler, early_stop


    # Training loop
    @staticmethod
    def train_loop(model, device, mixup, criterion, optimizer, scheduler, scaler, train_loader, loop):

        model.train()
        running_loss = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            
            # Mixup
            if mixup is not None:
                X, y1, y2 = mixup.make_batch(X, y)
                X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
            else:
                X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            if scaler is not None: # amp
                with amp.autocast():
                    y_hat = model(X)
                    if mixup is not None:
                        loss  = mixup.lamda * criterion(y_hat, y1) + (1 - mixup.lamda) * criterion(y_hat, y2)
                    else: # No mixup
                        loss = criterion(y_hat, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else: # no amp
                y_hat = model(X)
                if mixup is not None:
                    loss  = mixup.lamda * criterion(y_hat, y1) + (1 - mixup.lamda) * criterion(y_hat, y2)
                else:
                    loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            
            # Update
            if scheduler is not None: scheduler.step()
            running_loss += loss.item()
            loop.set_description(f"Training Batch [{batch_idx + 1} / {len(train_loader)}: loss = {round(loss.item(), 4)}]")

        running_loss /= len(train_loader)
        
        lr = scheduler.get_last_lr() if scheduler is not None else optimizer.param_groups[0]['lr']

        return running_loss, lr

    
    # Validation loop
    @staticmethod
    def validation_loop(model, device, criterion, scaler, val_loader, loop):

        model.eval()
        running_loss   = 0
        targets, preds = [], []

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                X = X.to(device)
                y = y.to(device)

                if scaler is not None:
                    with amp.autocast():
                        y_hat = model(X)
                else:
                    y_hat = model(X)

                loss  = criterion(y_hat, y)
                running_loss += loss.item()
                targets.extend(y.cpu().numpy().tolist())
                preds.extend(np.concatenate(y_hat.cpu().numpy()).tolist())
                loop.set_description(f"Validation Batch [{batch_idx + 1} / {len(val_loader)}: loss = {round(loss.item(), 4)}]")
                
        running_loss /= len(val_loader)
        val_auc = roc_auc_score(targets, preds)

        return running_loss, val_auc, preds

    
    # Save training process
    def make_checkpoint(self, model, optimizer, scheduler, scaler, epoch, val_loss, val_auc, val_predictions):

        sched_dict  = scheduler.state_dict() if scheduler is not None else None
        scaler_dict = scaler.state_dict() if scaler is not None else None
        
        checkpoint = {'model'     : model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'scheduler' : sched_dict,
                      'scaler'    : scaler_dict,
                      'epoch'     : epoch,
                      'val_loss'  : val_loss,
                      'val_auc'   : val_auc}

        torch.save(checkpoint, self.chekpoint_dirname)
        np.save(self.pred_dirname, np.array(val_predictions))
    
        return
    
    
    @staticmethod
    def load_checkpoint(config, model, optimizer, scheduler, scaler):
        
        print('Loading checkpoint')
        checkpoint = torch.load(config['load_checkpoint'])
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if config['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        if config['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            
        epoch    = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        val_auc  = checkpoint['val_auc']
            
        return model, optimizer, scheduler, scaler, epoch, val_loss, val_auc
    
    # Train
    def __call__(self, config, fold):
        
        # Setup
        device, mixup, model, criterion, scaler, optimizer, train_loader, val_loader, scheduler, early_stop = self.setup_train(config, fold)
        
        # Load checkpoint
        if config['load_checkpoint'] is not None:
            model, optimizer, scheduler, scaler, epoch, val_loss, val_auc = self.load_checkpoint(config, model, optimizer, scheduler, scaler)
        else:
            cur_epoch = 0
        
        # Setup loop
        loop      = tqdm(range(cur_epoch, config['no_epochs']))
        stats     = {"epoch":[], "train_loss":[], "val_loss":[], "val_auc": [], "lr": []}
        model.to(device);

        for epoch in loop:

            train_loss, train_lr     = self.train_loop(model, device, mixup, criterion, optimizer, scheduler, scaler, train_loader, loop)
            val_loss, val_auc, preds = self.validation_loop(model, device, criterion, scaler, val_loader, loop)
            
            # Early stopping and checkpointing
            if config['early_stop']:
                early_stop(val_loss)
                if early_stop.early_stop: break
            
            if config['early_stop']:
                if early_stop.save and self.save: 
                    self.make_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, val_auc, preds)
            else:
                if self.save:
                    self.make_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, val_auc, preds)
            
            # Append statistics
            stats["epoch"].append(epoch)
            stats["train_loss"].append(train_loss)
            stats["lr"].append(train_lr)
            stats["val_loss"].append(val_loss)
            stats["val_auc"].append(val_auc)

            # Update progress bar
            if config['early_stop']:
                loop.set_postfix(train_loss = train_loss, val_loss = val_loss, val_auc = val_auc, early_stopping = early_stop.counter)
            else:
                loop.set_postfix(train_loss = train_loss, val_loss = val_loss, val_auc = val_auc)
            
        return stats
    
    def plot(self, stats, fold):
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 3))

        ax1.plot(stats['epoch'], stats['val_loss'])
        ax1.plot(stats['epoch'], stats['train_loss'])
        ax1.legend(['Val loss', 'Train loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(stats['epoch'], stats['val_auc'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val ROC - AUC')
        fig.suptitle('Fold ' + str(fold));
        plt.show()
        
        if self.save:
            fig.savefig(self.plot_dirname)
        
        return
