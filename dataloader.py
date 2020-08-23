import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter2index, add_sos=True):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    result = []
    for i in transcript:
        indice = []
        if add_sos:
            indice.append(letter2index['<sos>']) # add <sos>
        for word in i:
            indice.extend([letter2index[letter] for letter in word.decode()]) # letter to index
            indice.append(letter2index[' ']) # add space between words
        indice.pop()
        indice.append(letter2index['<eos>']) # add <eos>
        result.append(indice)

    return result


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = {value: index for index, value in enumerate(letter_list)}
    index2letter = {index: value for index, value in enumerate(letter_list)}
    return letter2index, index2letter

def transform_index_to_letter(index, letter2index, index2letter):
    result = []
    for i in index:
        transcript = ''
        for j in i:
            if j == letter2index['<eos>']:
                break
            else:
                transcript += (index2letter[j])
        result.append(transcript)
    return result
            
class Speech2TextDataset(Dataset):
    '''
    size:
    speech: N * T * X
    text: N * T
    
    N: batch_size
    T: variable sequential length
    X: data dimension/channel
    '''
    def __init__(self, speech, text=None):
        self.speech = speech
        self.isTrain = False
        if text is not None:
            self.isTrain = True
            letter2index, index2letter = create_dictionaries(LETTER_LIST)
            text = transform_letter_to_index(text, letter2index)
        self.text = text
        print("Dataset in Training mode: ", self.isTrain)


    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))

def collate_train(seq_list):
    '''
    return:
        padded_inputs - padded speech
        input_lengths - length of each speech
        padded_targets - padded text
        target_lengths - length of each text
    '''
    inputs, targets = zip(*seq_list)
    input_lengths = torch.LongTensor([x.shape[0] for x in inputs])
    padded_inputs = rnn.pad_sequence([torch.Tensor(i) for i in inputs], batch_first=True)
    target_lengths = torch.LongTensor([y.shape[0] for y in targets])
    padded_targets = rnn.pad_sequence([torch.LongTensor(i) for i in targets], batch_first=True)
    return padded_inputs, input_lengths, padded_targets, target_lengths


def collate(seq_list):
    '''
    return:
        padded_inputs - padded speech
        input_lengths - length of each speech
    '''
    inputs = seq_list
    input_lengths = torch.LongTensor([x.shape[0] for x in inputs])
    padded_inputs = rnn.pad_sequence([torch.Tensor(i) for i in inputs], batch_first=True)
    return padded_inputs, input_lengths

# data loader

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, length, batch_size):
        letter2index, index2letter = create_dictionaries(LETTER_LIST)
        dataset = transform_letter_to_index(dataset, letter2index, add_sos=False)
        self.dataset = dataset
        self.length = length
        self.batch_size = batch_size
        total_length = sum([len(i) for i in dataset])
        self.num_batch = int(np.ceil(np.floor(total_length / self.length) / self.batch_size))

    def __iter__(self):
        np.random.shuffle(self.dataset)
        concat_dataset = np.concatenate(self.dataset)
        concat_dataset = concat_dataset[:concat_dataset.shape[0]-(concat_dataset.shape[0]%self.length)+1]
        datas = concat_dataset[:-1]
        labels = concat_dataset[1:]
        
        datas = datas.reshape(-1, self.length)
        labels = labels.reshape(-1, self.length)
        
        for i in range(self.num_batch):
            yield torch.LongTensor(datas[i*self.batch_size:(i+1)*self.batch_size, :]), F.pad(torch.LongTensor(labels[i*self.batch_size:(i+1)*self.batch_size, :]), (1,0,0,0), value=33)
                   
