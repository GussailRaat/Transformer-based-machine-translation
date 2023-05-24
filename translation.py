import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
import pickle
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time, random, math, string
# import spacy
import numpy as np
import math
# import cv2
import sys
import os
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
# ------------------------------------------
SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# ------------------------------------------
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
import pickle
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time, random, math, string
from tokenizers import ByteLevelBPETokenizer

def generate_text_for_tokenizer(path):
    with open(path + 'en.txt', "w") as engText, open(path + 'de.txt', "w") as gerText:
        with open(path + 'train.en', 'r') as fileSRC, open(path + 'train.de', 'r') as fileTRG:
            for line in zip(fileSRC, fileTRG):
                engText.write(line[0].replace('\n', '').strip().lower() + '\n')
                gerText.write(line[1].replace('\n', '').strip().lower() + '\n')
        # ----------------------------------------------------------------------------------
        with open(path + 'valid.en', 'r') as fileSRC, open(path + 'valid.de', 'r') as fileTRG:
            for line in zip(fileSRC, fileTRG):
                engText.write(line[0].replace('\n', '').strip().lower() + '\n')
                gerText.write(line[1].replace('\n', '').strip().lower() + '\n')
        # ----------------------------------------------------------------------------------
        with open(path + 'test.en', 'r') as fileSRC, open(path + 'test.de', 'r') as fileTRG:
            for line in zip(fileSRC, fileTRG):
                engText.write(line[0].replace('\n', '').strip().lower() + '\n')
                gerText.write(line[1].replace('\n', '').strip().lower() + '\n')

def getDataset(path, src, trg):
    dataset = {}
    with open(path + 'train.'+src+'', 'r') as fileSRC, open(path + 'train.'+trg+'', 'r') as fileTRG:
        data, dataSRC, dataTRG = {}, [], []
        for line in zip(fileSRC, fileTRG):
            dataSRC.append(line[0].replace('\n', '').strip().lower())
            dataTRG.append(line[1].replace('\n', '').strip().lower())
        data['src'] = dataSRC
        data['trg'] = dataTRG
        dataset['train'] = data
    # -------------------------------------------
    with open(path + 'valid.'+src+'', 'r') as fileSRC, open(path + 'valid.'+trg+'', 'r') as fileTRG:
        data, dataSRC, dataTRG = {}, [], []
        for line in zip(fileSRC, fileTRG):
            dataSRC.append(line[0].replace('\n', '').strip().lower())
            dataTRG.append(line[1].replace('\n', '').strip().lower())
        data['src'] = dataSRC
        data['trg'] = dataTRG
        dataset['valid'] = data
    # --------------------------------------------
    with open(path + 'test.'+src+'', 'r') as fileSRC, open(path + 'test.'+trg+'', 'r') as fileTRG:
        data, dataSRC, dataTRG = {}, [], []
        for line in zip(fileSRC, fileTRG):
            dataSRC.append(line[0].replace('\n', '').strip().lower())
            dataTRG.append(line[1].replace('\n', '').strip().lower())
        data['src'] = dataSRC
        data['trg'] = dataTRG
        dataset['test'] = data
    return dataset

def getTokenizers(pathSrcText, pathTrgText, vocab_size, min_frequency):
    tokenizerSRC = ByteLevelBPETokenizer()
    tokenizerSRC.train(pathSrcText,
                      vocab_size=vocab_size,
                      min_frequency=min_frequency,
                      special_tokens=["<sos>", "<eos>", "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
                     )

    tokenizerTRG = ByteLevelBPETokenizer()
    tokenizerTRG.train(pathTrgText,
                      vocab_size=vocab_size,
                      min_frequency=min_frequency,
                      special_tokens=["<sos>", "<eos>", "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
                     )
    return tokenizerSRC, tokenizerTRG

def collate_fn(batch):
    # # Sort the batch in descending order of sequence length
    # batch.sort(key=lambda x: len(x), reverse=False)

    # Convert the batch elements to tensors
    src, trg = [], []
    for sample in batch:
        interSRC, interTRG = [], []
        for sampleSRC, sampleTRG in zip(sample['src'], sample['trg']):
            if isinstance(sampleSRC, str) or isinstance(sampleTRG, str):
                print('Please give numeric values not str')
                return
            else:
                interSRC.append(int(sampleSRC))
                interTRG.append(int(sampleTRG))
        src.append(interSRC)
        trg.append(interTRG)

    lenSRC, lenTRG = zip(*[(len(x['src']), len(x['trg'])) for x in batch])

    # Pad the sequences in the batch to have the same length
    paddedSRC, paddedTRG = [], []
    for sampleSRC, sampleTRG in zip(src, trg):
        padded_sample_SRC = sampleSRC + [tokenizer['src'].token_to_id('<pad>')] * (max(lenSRC) - len(sampleSRC))
        padded_sample_TRG = sampleTRG + [tokenizer['trg'].token_to_id('<pad>')] * (max(lenTRG) - len(sampleTRG))
        paddedSRC.append(padded_sample_SRC)
        paddedTRG.append(padded_sample_TRG)

    # print(torch.tensor(paddedSRC).shape, torch.tensor(paddedTRG).shape)
    return {'src': torch.tensor(paddedSRC), 'trg': torch.tensor(paddedTRG)}


class myDataset(Dataset):
    def __init__(self, data, tokenizerSRC, tokenizerTRG, max_length):
        self.data = data
        self.tokenizerSRC = tokenizerSRC
        self.tokenizerTRG = tokenizerTRG
        self.max_length  = max_length

    def __preprocess__(self, text, tokenizer):
        text = text.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        text = "<sos> " + text + " <eos>"
        enc = tokenizer.encode(text, add_special_tokens=True)
        enc_ids = torch.tensor(enc.ids)
        return enc_ids

    def __getitem__(self, index):
        src = self.__preprocess__(self.data['src'][index], self.tokenizerSRC)
        trg = self.__preprocess__(self.data['trg'][index], self.tokenizerTRG)
        return {'src': src, 'trg': trg}

    def __len__(self):
        return len(self.data['src'])

# # ===============================================
# path = '/DATA/dushyant_1821cs17/experiments/paper_work/current/droplier/multimodal/translation/data1/multi30k/'
# generate_text_for_tokenizer(path) # to gernerate 'englishText.txt' and 'germanText.txt'
# # ----------------------------------------------------------------------------------------------
# pathSrcText = path + 'englishText.txt'
# pathTrgText = path + 'germanText.txt'
# # ----------------------------------------------------------------------------------------------
# tokenizer['src'], tokenizer['trg'] = getTokenizers(pathSrcText, pathTrgText, vocab_size=5000, min_frequency=1)
# # ----------------------------------------------------------------------------------------------
# rawDataset     = getDataset(path)
# train_dataset  = myDataset(rawDataset['train'], tokenizer['src'], tokenizer['trg'], max_length=25)
# train_iterator = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
#
# for i in train_iterator:
#     print(i['trg'], '\n')
#     print(i['trg_shift'])
#     break

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        #x = [batch size, seq len, hid dim]

        return x

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) #src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        #trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool() #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask #trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len], trg = [batch size, trg len]
        src_mask = self.make_src_mask(src) #src_mask = [batch size, 1, 1, src len]
        trg_mask = self.make_trg_mask(trg) #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask) #enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask) #output = [batch size, trg len, output dim], attention = [batch size, n heads, trg len, src len]
        return output, attention

################################################################################
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    with tqdm(iterator, unit="batch") as tLoader:
        for i, batch in enumerate(tLoader):
            if torch.cuda.is_available():
                src = batch['src'].to(device)
                trg = batch['trg'].to(device)

                optimizer.zero_grad()
                output_pred, _ = model(src, trg[:, :-1])

                # Flatten the output sequence and the target sequence for calculating the loss
                output_pred_flat = output_pred.view(-1, output_pred.size(-1))
                target_seq_flat = trg[:, 1:].contiguous().view(-1)

                loss = criterion(output_pred_flat, target_seq_flat)
                loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
    return train_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    test_loss = 0
    with tqdm(iterator, unit="batch") as tLoader:
        for i, batch in enumerate(tLoader):
            if torch.cuda.is_available():
                src = batch['src'].to(device)
                trg = batch['trg'].to(device)

                optimizer.zero_grad()
                output_pred, _ = model(src, trg[:, :-1])

                output_pred_flat = output_pred.view(-1, output_pred.size(-1))
                target_seq_flat = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output_pred_flat, target_seq_flat)
                test_loss += loss.item()

    return test_loss / len(iterator)


def translate_sentence(model, src, tokenizer, device, max_length):
    model.eval()

    # src_mask = data['src_mask'].unsqueeze(0).unsqueeze(1).unsqueeze(1)
    src_mask = model.make_src_mask(src).to(device)
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    trg_indexes = [tokenizer.token_to_id('<sos>')]

    # Autoregressively generate the remaining tokens in the output sequence
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == tokenizer.token_to_id('<eos>'):
            break
    return torch.tensor(trg_indexes), attention

def evaluate_bleu(model, dataset, tokenizer, device, max_length):
    trgs = []
    pred_trgs = []

    with tqdm(dataset, unit="batch") as tDataset:
        for data in tDataset:
            src = data['src'].unsqueeze(0).to(device)
            trg = data['trg'].to(device)

            pred_trg, _ = translate_sentence(model, src, tokenizer, device, max_length)

            # convert token into words
            pred_trg = tokenizer.decode(pred_trg.tolist()).lower().strip().split(' ')
            trg = tokenizer.decode(trg.tolist()).lower().strip().split(' ')

            # Please note that No need with BPE
            # # cut off <eos> token
            # pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

    return bleu_score(pred_trgs, trgs)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

########################################################################################
batch_size  = 128
num_workers = 0
max_length  = 128
N_EPOCHS    = 100
clip        = 1
LEARNING_RATE = 0.0005
earlyStop   = 100
vocab_size = 30000
min_frequency = 2
# ======================================================================================
# define src and trg
src, trg = str(sys.argv[1]), str(sys.argv[2])
print(src, trg)
# ======================================================================================

# ======================================================================================
# Get dataset
# ======================================================================================
path = 'data1/multi30k/'
generate_text_for_tokenizer(path) # to gernerate 'englishText.txt' and 'germanText.txt'
pathSrcText = path + src +'.txt'
pathTrgText = path + trg + '.txt'
# ----------------------------------------------------------------------------------------------
tokenizer = {}
tokenizer['src'], tokenizer['trg'] = getTokenizers(pathSrcText, pathTrgText, vocab_size=vocab_size, min_frequency=min_frequency)
src_vocab_size, trg_vocab_size = len(tokenizer['src'].get_vocab()), len(tokenizer['trg'].get_vocab())
# ----------------------------------------------------------------------------------------------
rawDataset     = getDataset(path, src, trg)
train_dataset  = myDataset(rawDataset['train'], tokenizer['src'], tokenizer['trg'], max_length=max_length)
valid_dataset  = myDataset(rawDataset['valid'], tokenizer['src'], tokenizer['trg'], max_length=max_length)
test_dataset   = myDataset(rawDataset['test'], tokenizer['src'], tokenizer['trg'], max_length=max_length)
# ----------------------------------------------------------------------------------------------
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
valid_iterator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
test_iterator  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
# ----------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# ----------------------------------------------------------------------------------------------

INPUT_DIM = src_vocab_size
OUTPUT_DIM = trg_vocab_size
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              max_length)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              max_length)

SRC_PAD_IDX = tokenizer['src'].token_to_id('<pad>')
TRG_PAD_IDX = tokenizer['trg'].token_to_id('<pad>')

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
# ----------------------------------------------------------------------------------------------

# We can check the number of parameters, noticing it is significantly less than the 37M for the convolutional sequence-to-sequence model.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# The paper does not mention which weight initialization scheme was used, however Xavier uniform seems to be common amongst Transformer models, so we use it here.
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# print(tokenizer['trg'].token_to_id('<sos>'), tokenizer['src'].token_to_id('<sos>'))
# print(tokenizer['trg'].token_to_id('<eos>'), tokenizer['src'].token_to_id('<eos>'))
# print(tokenizer['trg'].token_to_id('<pad>'), tokenizer['src'].token_to_id('<pad>'))

best_valid_PP = float('inf')
savedAtEpoch = 0
fPath = str(sys.argv[1]) +'_'+ str(sys.argv[2]) +'_'+ str(batch_size) +'_'+ str(max_length)
weight_path = 'weights/'+fPath+'.pt'
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')

    valid_PP = math.exp(valid_loss)
    if valid_PP < best_valid_PP:
        savedAtEpoch = epoch
        best_valid_PP = valid_PP
        torch.save(model.state_dict(), weight_path)
    elif (epoch - savedAtEpoch) >= earlyStop:
        break
    else:
        print('\n')

# =============================================================================
# Inference (Testing phase)
# =============================================================================
model.load_state_dict(torch.load(weight_path))
test_loss = evaluate(model, test_iterator, criterion)
b_score   = evaluate_bleu(model, test_dataset, tokenizer['trg'], device, max_length)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU score: {b_score*100:7.3f} ')

open('results/'+sys.argv[0].split('.')[0]+'_'+ str(sys.argv[1]) +'_'+ str(sys.argv[2])+'.txt', 'a').write(fPath+'\n'+
                                                             'savedAtEpoch: '+ str(savedAtEpoch) + '\t||\t'+
                                                             'loss: '+ str(test_loss) + '\t||\t'+
                                                             'PPL: '+ str(math.exp(test_loss)) + '\t||\t'+
                                                             'BLEU: '+ str(b_score*100) +'\n'+
                                                             '==========================================\n\n')
