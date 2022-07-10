import torch
from torch import nn
import sys
from src import models
from src import cont
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import Precision_score, mER_score
from src.eval_metrics import *




def get_Cont_module(hyp_params):
    a2l_module = getattr(Cont, 'ContModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(Cont, 'ContModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MUL':
        Cont_criterion = None
        Cont_a2l_module, Cont_v2l_module = None, None
        Cont_a2l_optimizer, Cont_v2l_optimizer = None, None
    else:
        from warpCont_pytorch import ContLoss
        Cont_criterion = ContLoss()
        Cont_a2l_module, Cont_v2l_module = get_Cont_module(hyp_params)
        if hyp_params.use_cuda:
            Cont_a2l_module, Cont_v2l_module = Cont_a2l_module.cuda(), Cont_v2l_module.cuda()
        Cont_a2l_optimizer = getattr(optim, hyp_params.optim)(Cont_a2l_module.parameters(), lr=hyp_params.lr)
        Cont_v2l_optimizer = getattr(optim, hyp_params.optim)(Cont_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'Cont_a2l_module': Cont_a2l_module,
                'Cont_v2l_module': Cont_v2l_module,
                'Cont_a2l_optimizer': Cont_a2l_optimizer,
                'Cont_v2l_optimizer': Cont_v2l_optimizer,
                'Cont_criterion': Cont_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)




def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    Cont_a2l_module = settings['Cont_a2l_module']
    Cont_v2l_module = settings['Cont_v2l_module']
    Cont_a2l_optimizer = settings['Cont_a2l_optimizer']
    Cont_v2l_optimizer = settings['Cont_v2l_optimizer']
    Cont_criterion = settings['Cont_criterion']
    
    scheduler = settings['scheduler']
    

    def train(model, optimizer, criterion, Cont_a2l_module, Cont_v2l_module, Cont_a2l_optimizer, Cont_v2l_optimizer, Cont_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, context1, context2, context3, context4 = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if Cont_criterion is not None:
                Cont_a2l_module.zero_grad()
                Cont_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    context1, context2, context3, context4, eval_attr = context1.cuda(), context2.cuda(), context3.cuda(), context4.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'EMOTIC':
                        eval_attr = eval_attr.long()
            
            batch_size = context1.size(0)
            batch_chunk = hyp_params.batch_chunk
            
          
            if Cont_criterion is not None:
                Cont_a2l_net = nn.DataParallel(Cont_a2l_module) if batch_size > 10 else Cont_a2l_module
                Cont_v2l_net = nn.DataParallel(Cont_v2l_module) if batch_size > 10 else Cont_v2l_module

                context2, a2l_position = Cont_a2l_net(context2) 
                context3, v2l_position = Cont_v2l_net(context3)
              
              
                position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                
                c1_length = torch.tensor([c1_len]*batch_size).int().cpu()
               
                c2_length = torch.tensor([c2_len]*batch_size).int().cpu()
                c3_length = torch.tensor([c3_len]*batch_size).int().cpu()
                c4_length = torch.tensor([c4_len]*batch_size).int().cpu()
                Cont_a2l_loss = Cont_criterion(a2l_position.transpose(0,1).cpu(), position, c2_length, c3_length)
                Cont_v2l_loss = Cont_criterion(v2l_position.transpose(0,1).cpu(), position, c3_length, c2_length)
                Cont_loss = Cont_a2l_loss + Cont_v2l_loss
                Cont_loss = Cont_loss.cuda() if hyp_params.use_cuda else Cont_loss
            else:
                Cont_loss = 0
        
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                context1_chunks = context1.chunk(batch_chunk, dim=0)
                context2_chunks = context2.chunk(batch_chunk, dim=0)
                context3_chunks = context3.chunk(batch_chunk, dim=0)
                context4_chunks = context4.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    context1_i, context2_i, context3_i,context4_i = context1_chunks[i], context2_chunks[i], context3_chunks[i], context4_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(context1_i, context2_i, context3_i, context4_i)
                    
                    if hyp_params.dataset == 'EMOTIC':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                Cont_loss.backward()
                combined_loss = raw_loss + Cont_loss
            else:
                preds, hiddens = net(context1, context2, context3, context4)
                if hyp_params.dataset == 'HECO':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + Cont_loss
                combined_loss.backward()
            
            if Cont_criterion is not None:
                torch.nn.utils.clip_grad_norm_(Cont_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(Cont_v2l_module.parameters(), hyp_params.clip)
                Cont_a2l_optimizer.step()
                Cont_v2l_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, Cont_a2l_module, Cont_v2l_module, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, context1, context2, context3, context4 = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        context1, context2, context3, context4 eval_attr = context1.cuda(), context2.cuda(), context3.cuda(), context4.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'EMOTIC':
                            eval_attr = eval_attr.long()
                        
                batch_size = context1.size(0)
                
                if (Cont_a2l_module is not None) and (Cont_v2l_module is not None):
                    Cont_a2l_net = nn.DataParallel(Cont_a2l_module) if batch_size > 10 else Cont_a2l_module
                    Cont_v2l_net = nn.DataParallel(Cont_v2l_module) if batch_size > 10 else Cont_v2l_module
                    context2, _ = Cont_a2l_net(context2)     
                    context3, _ = Cont_v2l_net(context3)   
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(context1, context2, context3, context4)
                if hyp_params.dataset == 'HECO':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

               
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train(model, optimizer, criterion, Cont_a2l_module, Cont_v2l_module, Cont_a2l_optimizer, Cont_v2l_optimizer, Cont_criterion)
        val_loss, _, _ = evaluate(model, Cont_a2l_module, Cont_v2l_module, criterion, test=False)
        test_loss, _, _ = evaluate(model, Cont_a2l_module, Cont_v2l_module, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)   

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, Cont_a2l_module, Cont_v2l_module, criterion, test=True)

    if hyp_params.dataset == "EMOTIC":
        eval_EMOTIC(results, truths, True)
    elif hyp_params.dataset == 'HECO':
        eval_HECO(results, truths, True)
    elif hyp_params.dataset == 'GroupWalk':
        eval_GroupWalk(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
