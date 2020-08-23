import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import Levenshtein as LS
from tqdm import tqdm
import logging
import os
import sys
import pandas as pd
import argparse

from dataloader import transform_letter_to_index, create_dictionaries, transform_index_to_letter, Speech2TextDataset, LanguageModelDataLoader, collate_train, collate
from models import Seq2Seq

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pre_train(model, train_loader, vali_loader, criterion, args, logger, letter2index, index2letter):
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr, weight_decay=5e-4)

    model.train()
    model.to(DEVICE)
    train_loss_ = []
    len_train_loader = train_loader.num_batch

    for epoch in range(args.pre_trained_epochs):
        # 1) Iterate through your loader
        train_loss = 0.0
        num_data = 0.0
        for batch_id, (x, y) in tqdm(enumerate(train_loader), total=len_train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            key = torch.zeros([x.size(0), 128]).to(DEVICE)
            values = torch.zeros([x.size(0), x.size(1), 128]).to(DEVICE)
            lengths = torch.zeros([x.size(0)]).to(DEVICE)
            preds = model.decoder(key, values, lengths, text=y, isTrain=True, isAttended=False)[0]

            loss = criterion(preds.view(-1, preds.size(2)), y[:,1:].contiguous().view(-1)).mean() # no <sos>

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * x.size(0)
            num_data += x.size(0)

        train_loss = train_loss / num_data
        train_loss_.append(train_loss)
        
        print('Epoch %i, train loss: %f'%(epoch+1, train_loss))

def vali_inference(model, vali_loader, letter2index, index2letter):
    model.eval()
    LS_dist = 0.0
    num_data = 0.0
    for batch_id, (x, x_length, y, y_length) in enumerate(vali_loader):
        x = x.to(DEVICE)
        x_length = x_length.to(DEVICE)
        
        preds = model(x, x_length, isTrain=False)
        
        preds_transcript = transform_index_to_letter(preds.argmax(-1).detach().cpu().numpy(), letter2index, index2letter)
        
        y_transcript = transform_index_to_letter(y[:,1:].numpy(), letter2index, index2letter) # note y is: <sos> ... <eos>

        for i in range(y.size(0)):
            LS_dist += LS.distance(preds_transcript[i], y_transcript[i])
        
        num_data += x.size(0)
    LS_dist /= num_data
    
    return LS_dist

def train(model, train_loader, vali_loader, criterion, args, logger, letter2index, index2letter):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler_p = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.pt_factor, patience=1, verbose=True)

    model.train()
    model.to(DEVICE)
    train_loss_ = []
    vali_loss_ = []
    len_train_loader = len(train_loader)

    for epoch in range(args.epochs):
        # 1) Iterate through your loader
        train_loss = 0.0
        num_data = 0.0
        for batch_id, (x, x_length, y, y_length) in tqdm(enumerate(train_loader), total=len_train_loader):
            model.train()
            # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
            # ignore

            # 3) Set the inputs to the device.
            x = x.to(DEVICE)
            x_length = x_length.to(DEVICE)
            y = y.to(DEVICE)
            y_length = y_length.to(DEVICE)

            # 4) Pass your inputs, and length of speech into the model.
            preds = model(x, x_length, y)[0]

            # 5) Generate a mask based on the lengths of the text to create a masked loss. 
            # 5.1) Ensure the mask is on the device and is the correct shape.
            mask = y[:, 1:] > 0 # note y is <sos> ... <eos>

            # 6) If necessary, reshape your predictions and origianl text input 
            # 6.1) Use .contiguous() if you need to. 
            preds = preds.view(-1, preds.size(-1))
            y = y[:, 1:].contiguous().view(-1) # note y is <sos> ... <eos>

            # 7) Use the criterion to get the loss.
            loss = criterion(preds, y) # no <sos>

            # 8) Use the mask to calculate a masked loss. 
            masked_loss = (loss * mask.view(-1)).sum()

            # 9) Run the backward pass on the masked loss. 
            masked_loss.backward()

            # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            torch.nn.utils.clip_grad_norm(model.parameters(), 2)

            # 11) Take a step with your optimizer
            optimizer.step()

            # 12) Normalize the masked loss
            masked_loss = masked_loss / mask.sum()
            train_loss += masked_loss.item() * x.size(0)
            num_data += x.size(0)

            # 13) Optionally print the training loss after every N batches
            if batch_id > 1 and batch_id % (int(len_train_loader/2)+1) == 0:
                vali_loss = vali_inference(model, vali_loader, letter2index, index2letter)
                print('train loss: %f, vali loss: %f'%(train_loss/num_data, vali_loss))

        train_loss = train_loss / num_data
        train_loss_.append(train_loss)
        vali_loss = vali_inference(model, vali_loader, letter2index, index2letter)
        vali_loss_.append(vali_loss)
        
        scheduler_p.step(vali_loss)
        
        torch.save(model.state_dict(), os.path.join(args.save_path,args.model_name,"model_%i.t7"%epoch))
        np.save(os.path.join(args.save_path, args.model_name, "train_loss_.npy"), train_loss_)
        np.save(os.path.join(args.save_path, args.model_name, "vali_loss_.npy"), vali_loss_)
        logger.info('Epoch %i, train loss: %f, vali loss: %f'%(epoch+1, train_loss, vali_loss))


###########################
#        argument         #
###########################

parser = argparse.ArgumentParser(description='Hw4p2')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--epochs", type=int, default=50, help='training epochs')
parser.add_argument("--batch_size", type=int, default=100, help='batch size')
parser.add_argument("--pre_trained_epochs", type=int, default=20, help='pre-training epochs')
parser.add_argument("--pre_trained_batch_size", type=int, default=3000, help='pre-training batch size')
parser.add_argument("--batch_size_test", type=int, default=300, help='batch size for test')
parser.add_argument("--pt_factor", type=float, default=0.5, help='plateau scheduler reduction factor')
parser.add_argument("--name", type=str, default='hw4p2', help='model name')
parser.add_argument("--data_path", type=str, help='data path')
parser.add_argument("--save_path", type=str, help='where to save')

args = parser.parse_args()

###########################
#        logger           #
###########################
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not os.path.isdir(os.path.join(args.save_path, args.model_name)):
    os.makedirs(os.path.join(args.save_path, args.model_name))

if os.path.isfile(os.path.join(args.save_path, args.model_name, 'logfile.log')):
    os.remove(os.path.join(args.save_path, args.model_name, 'logfile.log'))
    
file_log_handler = logging.FileHandler(os.path.join(args.save_path, args.model_name, 'logfile.log'))
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)

attrs = vars(args)
for item in attrs.items():
    logger.info("%s: %s"%item)


###########################
#        Main             #
###########################

# Data loading
letter2index, index2letter = create_dictionaries(LETTER_LIST)

speech_train = np.load(os.path.join(args.data_path, 'train_new.npy'), allow_pickle=True, encoding='bytes')
transcript_train = np.load(os.path.join(args.data_path, 'train_transcripts.npy'), allow_pickle=True,encoding='bytes')

speech_valid = np.load(os.path.join(args.data_path, 'dev_new.npy'), allow_pickle=True, encoding='bytes')
transcript_valid = np.load(os.path.join(args.data_path, 'dev_transcripts.npy'), allow_pickle=True,encoding='bytes')

speech_test = np.load(os.path.join(args.data_path, 'test_new.npy'), allow_pickle=True, encoding='bytes')

train_dataset = Speech2TextDataset(speech_train, text=transcript_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train, num_workers=os.cpu_count())

vali_dataset = Speech2TextDataset(speech_valid, text=transcript_valid)
vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_train, num_workers=os.cpu_count())

test_dataset = Speech2TextDataset(speech_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, collate_fn=collate, num_workers=os.cpu_count())

pre_train_loader = LanguageModelDataLoader(transcript_train, length=50, batch_size=args.pre_trained_batch_size)

# Model Initialization
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(40, len(LETTER_LIST), 256, value_size=256, key_size=256).to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction='none').to(DEVICE)

# Pre-train
pre_train(model, pre_train_loader, pre_train_loader, criterion, args, logger, letter2index, index2letter)

# Train
train(model, train_loader, vali_loader, criterion, args, logger, letter2index, index2letter)

# Inference

# greedy search
result = []
result_probs = []
model.eval()
with torch.no_grad():
    for batch_id, (x, x_length) in enumerate(test_loader):
        x = x.to(DEVICE)
        x_length = x_length.to(DEVICE)

        preds, probs = model(x, x_length, isTrain=False, random=False)      
        preds_transcript = transform_index_to_letter(preds.argmax(-1).detach().cpu().numpy(), letter2index, index2letter)
        result.extend(preds_transcript)
        result_probs.append(probs)
        
result_probs = torch.cat(result_probs)
        
result_greedy = result
result_probs_greedy = result_probs    

# random search
result = []
result_probs = []
model.eval()
with torch.no_grad():
    for batch_id, (x, x_length) in enumerate(test_loader):
        x = x.to(DEVICE)
        x_length = x_length.to(DEVICE)

        random_result = []
        random_probs = []
        for i in tqdm(range(1000)):
            preds, probs = model(x, x_length, isTrain=False, random=True)      
            preds_transcript = transform_index_to_letter(preds.argmax(-1).detach().cpu().numpy(), letter2index, index2letter)
            random_result.append(preds_transcript)
            random_probs.append(probs)
        
        random_probs = torch.stack(random_probs)
        max_probs = random_probs.max(0)[0]
        max_index = random_probs.argmax(0)
        for i in range(len(max_index)):
            result.append(random_result[max_index[i]][i])
            
        result_probs.append(max_probs)
        
result_probs = torch.cat(result_probs)

result_random = result
result_probs_random = result_probs

# combine greedy search and random search
result = []
for i in range(len(result_random)):
    if result_probs_random[i] >= result_probs_greedy[i]:
        result.append(result_random[i])
    else:
        result.append(result_greedy[i])

# save result
f = pd.read_csv(os.path.join(args.data_path, 'test_sample_submission.csv'))
f['Predicted'] = result
f.to_csv(os.path.join(args.data_path, 'test_sample_submission.csv'), index=False)