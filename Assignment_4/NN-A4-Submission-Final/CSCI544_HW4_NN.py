#!/usr/bin/env python
# coding: utf-8

# In[10]:


# get_ipython().system('pip install datasets')
# get_ipython().system('pip install torch torchvision')
# get_ipython().system('pip install ipython-autotime')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gzip
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_dataset
from typing import List, Dict, Tuple
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')

# %load_ext autotime


# In[11]:


# config & constants
try:
    os.chdir("/content/drive/MyDrive/Colab Notebooks/CSCI544/hw4")
except:
    pass

class PathConfig:
    HW4_DIR = os.getcwd() 
    GLOVE_100d_File = os.path.join(HW4_DIR, "glove.6B.100d.gz")
    OUTPUT_DIR = os.path.join(HW4_DIR, "output")
    DATASET_PATH = os.path.join(HW4_DIR, "data")

    
    BILSTM_1_FILE_PATH = os.path.join(OUTPUT_DIR, "bilstm-1.pt")
    BILSTM_2_FILE_PATH = os.path.join(OUTPUT_DIR, "bilstm-2.pt")
    BILSTM_BONUS_FILE_PATH = os.path.join(OUTPUT_DIR, "bilstm-bonus.pt")

    DEV_1_FILE_PATH = os.path.join(OUTPUT_DIR, "dev1.out")
    DEV_2_FILE_PATH = os.path.join(OUTPUT_DIR, "dev2.out")
    DEV_BONUS_FILE_PATH = os.path.join(OUTPUT_DIR, "dev-bonus.out")
    
    TEST_1_FILE_PATH = os.path.join(OUTPUT_DIR, "test1.out")
    TEST_2_FILE_PATH = os.path.join(OUTPUT_DIR, "test2.out")
    TEST_BONUS_FILE_PATH = os.path.join(OUTPUT_DIR, "test-pred-bonus.out")


class DatasetConfig:
    cols = ["index", "words", "labels"]
    train_data_path = os.path.join(PathConfig.DATASET_PATH, "train")
    dev_data_path = os.path.join(PathConfig.DATASET_PATH, "dev")
    test_data_path = os.path.join(PathConfig.DATASET_PATH, "test")

    # NER Tags list and converter dictionaries
    ner_tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    ner_idx2tag = {v: k for k, v in ner_tag2idx.items()}

    # Constants
    EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 30
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 1
    LSTM_DROPOUT = 0.33
    LINEAR_OUTPUT_DIM = 128
    LEARNING_RATE = 0.015
    WEIGHT_DECAY = 1e-5
    
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 40
    MOMENTUM = 0.9

    # Early stopping
    PATIENCE = 5
    
    # Learning rate scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
    SCHEDULER_MIN_LR = 1e-6


if not os.path.exists(PathConfig.OUTPUT_DIR):
    os.makedirs(PathConfig.OUTPUT_DIR)


# In[12]:


def load_datasets(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    sentences, tags = [], []
    current_sentence, current_tags = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[1]
                tag = parts[2] if len(parts) >= 3 else 'O'
                current_sentence.append(word)
                current_tags.append(tag)
            elif current_sentence:
                sentences.append(current_sentence)
                tags.append(current_tags)
                current_sentence, current_tags = [], []
    
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    
    return sentences, tags


# In[42]:


class BLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_matrix=None):
        super(BLSTM_NER, self).__init__()
        
        # Embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, DatasetConfig.EMBEDDING_DIM)
            
        self.embedding_dropout = nn.Dropout(0.3)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=DatasetConfig.EMBEDDING_DIM,
            hidden_size=DatasetConfig.LSTM_HIDDEN_DIM,
            num_layers=DatasetConfig.LSTM_LAYERS,
            bidirectional=True,
            dropout=DatasetConfig.LSTM_DROPOUT if DatasetConfig.LSTM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.hidden1 = nn.Linear(DatasetConfig.LSTM_HIDDEN_DIM * 2, DatasetConfig.LINEAR_OUTPUT_DIM)
        self.batch_norm = nn.BatchNorm1d(DatasetConfig.LINEAR_OUTPUT_DIM)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(DatasetConfig.LINEAR_OUTPUT_DIM, tagset_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embedding if not pretrained
        if not isinstance(self.embedding, nn.Embedding) or self.embedding.weight.requires_grad:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        # Initialize linear layers
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)
    
    def forward(self, x):
        # Embedding with dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Additional processing
        hidden = self.hidden1(lstm_out)
        hidden = hidden.permute(0, 2, 1)  # For batch norm
        hidden = self.batch_norm(hidden)
        hidden = hidden.permute(0, 2, 1)  # Back to original shape
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        tag_space = self.hidden2tag(hidden)
        
        return tag_space


# In[43]:


class BLSTM_NER_with_Chars(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, num_chars: int, 
                 embedding_matrix: torch.Tensor = None):
        super(BLSTM_NER_with_Chars, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, DatasetConfig.EMBEDDING_DIM)
        if embedding_matrix is not None:
            self.word_embedding.weight.data.copy_(embedding_matrix)
            self.word_embedding.weight.requires_grad = True
        
        self.char_cnn = CharacterCNN(num_chars, DatasetConfig.CHAR_EMBEDDING_DIM)
        
        char_cnn_output_dim = 150  # 50 * 3 kernels
        self.lstm = nn.LSTM(
            input_size=DatasetConfig.EMBEDDING_DIM + char_cnn_output_dim,
            hidden_size=DatasetConfig.LSTM_HIDDEN_DIM,
            num_layers=DatasetConfig.LSTM_LAYERS,
            bidirectional=True,
            dropout=DatasetConfig.LSTM_DROPOUT if DatasetConfig.LSTM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        self.linear = nn.Linear(DatasetConfig.LSTM_HIDDEN_DIM * 2, DatasetConfig.LINEAR_OUTPUT_DIM)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(DatasetConfig.LINEAR_OUTPUT_DIM, num_tags)
    
    def forward(self, word_ids, char_ids):
        word_embeds = self.word_embedding(word_ids)
        char_embeds = self.char_cnn(char_ids)
        combined_embeds = torch.cat([word_embeds, char_embeds], dim=2)
        
        lstm_out, _ = self.lstm(combined_embeds)
        linear_out = self.elu(self.linear(lstm_out))
        return self.classifier(linear_out)


# In[35]:


# Dataset and DataLoader
class NERDataset(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], 
                 word2idx: Dict[str, int], case_sensitive: bool = False):
        self.sentences = []
        self.tags = []
        
        for sentence, sentence_tags in zip(sentences, tags):
            if case_sensitive:
                self.sentences.append([word2idx.get(word, word2idx['<UNK>']) 
                                    for word in sentence])
            else:
                self.sentences.append([word2idx.get(word.lower(), word2idx['<UNK>']) 
                                    for word in sentence])
            self.tags.append([DatasetConfig.ner_tag2idx[tag] for tag in sentence_tags])
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.tags[idx])



# In[44]:


# helper functions       
def pad_sequence(batch):
    sentences = [item[0] for item in batch]
    tags = [item[1] for item in batch]
    
    max_len = max(len(s) for s in sentences)
    
    padded_sentences = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    for i, (sentence, tag) in enumerate(zip(sentences, tags)):
        length = len(sentence)
        padded_sentences[i, :length] = sentence
        padded_tags[i, :length] = tag
        mask[i, :length] = 1
    
    return padded_sentences, padded_tags, mask


def pad_sequence_with_chars(batch):
    sentences = [item[0] for item in batch]
    char_ids = [item[1] for item in batch]
    tags = [item[2] for item in batch]
    
    max_len = max(len(s) for s in sentences)
    
    padded_sentences = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_char_ids = torch.zeros((len(batch), max_len, char_ids[0].size(1)), dtype=torch.long)
    padded_tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    for i, (sentence, chars, tag) in enumerate(zip(sentences, char_ids, tags)):
        length = len(sentence)
        padded_sentences[i, :length] = sentence
        padded_char_ids[i, :length] = chars
        padded_tags[i, :length] = tag
        mask[i, :length] = 1
    
    return padded_sentences, padded_char_ids, padded_tags, mask


# In[45]:


# 2. Dataset class for character-level features
class NERDatasetWithChars(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], 
                 word2idx: Dict[str, int], char2idx: Dict[str, int], 
                 case_sensitive: bool = True):
        self.sentences = []
        self.tags = []
        self.char_ids = []
        self.max_word_len = 20
        
        for sentence, sentence_tags in zip(sentences, tags):
            if case_sensitive:
                self.sentences.append([word2idx.get(word, word2idx['<UNK>']) for word in sentence])
            else:
                self.sentences.append([word2idx.get(word.lower(), word2idx['<UNK>']) for word in sentence])
            
            self.tags.append([DatasetConfig.ner_tag2idx[tag] for tag in sentence_tags])
            
            char_ids_sent = []
            for word in sentence:
                char_ids_word = [char2idx.get(c, char2idx['<UNK>']) for c in word[:self.max_word_len]]
                char_ids_word += [char2idx['<PAD>']] * (self.max_word_len - len(char_ids_word))
                char_ids_sent.append(char_ids_word)
            self.char_ids.append(char_ids_sent)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sentences[idx]), 
                torch.tensor(self.char_ids[idx]), 
                torch.tensor(self.tags[idx]))

class BLSTM_NER_with_Chars(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, num_chars: int, 
                 embedding_matrix: torch.Tensor = None):
        super(BLSTM_NER_with_Chars, self).__init__()
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, DatasetConfig.EMBEDDING_DIM)
        if embedding_matrix is not None:
            self.word_embedding.weight.data.copy_(embedding_matrix)
            self.word_embedding.weight.requires_grad = True
        
        # Character CNN
        self.char_cnn = CharacterCNN(num_chars, DatasetConfig.CHAR_EMBEDDING_DIM)
        
        # Calculate total input size for LSTM
        char_cnn_output_dim = 150  # This is the output size from CharacterCNN
        total_embedding_dim = DatasetConfig.EMBEDDING_DIM + char_cnn_output_dim
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=total_embedding_dim,  # Combined word and char embeddings
            hidden_size=DatasetConfig.LSTM_HIDDEN_DIM,
            num_layers=DatasetConfig.LSTM_LAYERS,
            bidirectional=True,
            dropout=DatasetConfig.LSTM_DROPOUT if DatasetConfig.LSTM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.linear = nn.Linear(DatasetConfig.LSTM_HIDDEN_DIM * 2, DatasetConfig.LINEAR_OUTPUT_DIM)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(DatasetConfig.LINEAR_OUTPUT_DIM, num_tags)
    
    def forward(self, word_ids, char_ids):
        word_embeds = self.word_embedding(word_ids)
        char_embeds = self.char_cnn(char_ids)
        combined_embeds = torch.cat([word_embeds, char_embeds], dim=2)
        
        lstm_out, _ = self.lstm(combined_embeds)
        linear_out = self.elu(self.linear(lstm_out))
        return self.classifier(linear_out)


# In[46]:


class CharacterCNN(nn.Module):
    def __init__(self, num_chars: int, char_embedding_dim: int = 30):
        super(CharacterCNN, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, char_embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, 50, kernel_size)
            for kernel_size in [3, 4, 5]
        ])
        
    def forward(self, x):
        batch_size, seq_len, max_word_len = x.shape
        x = self.char_embedding(x.view(-1, max_word_len))
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(conv_out.squeeze(2))
        
        x = torch.cat(conv_outputs, dim=1)
        return x.view(batch_size, seq_len, -1)


# In[47]:


# Build vocabularies
def build_vocab(sentences: List[List[str]], case_sensitive: bool = False) -> Dict[str, int]:
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    for sentence in sentences:
        for word in sentence:
            word_key = word if case_sensitive else word.lower()
            if word_key not in word2idx:
                word2idx[word_key] = len(word2idx)
    
    return word2idx

# vocab for bonus
def build_char_vocab(sentences: List[List[str]]) -> Dict[str, int]:
    char2idx = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                if char not in char2idx:
                    char2idx[char] = len(char2idx)
    return char2idx


# In[20]:


def initialize_embeddings(word2idx: Dict[str, int], glove_path: str = None) -> torch.Tensor:
    """
    Initialize embeddings for both tasks
    Args:
        word2idx: Vocabulary dictionary
        glove_path: Path to GloVe embeddings (only for Task 2)
    """
    vocab_size = len(word2idx)
    
    if glove_path is None:
        # Task 1: Random initialization
        return torch.randn((vocab_size, DatasetConfig.EMBEDDING_DIM)) * 0.1
    
    # Task 2: Initialize with GloVe
    embedding_matrix = torch.randn((vocab_size, DatasetConfig.EMBEDDING_DIM)) * 0.1
    
    try:
        with gzip.open(glove_path, 'rt', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.FloatTensor([float(x) for x in values[1:]])
                
                # For exact matches
                if word in word2idx:
                    embedding_matrix[word2idx[word]] = vector
                
                # For case variations
                word_lower = word.lower()
                for vocab_word in word2idx:
                    if vocab_word.lower() == word_lower:
                        idx = word2idx[vocab_word]
                        if vocab_word.isupper():
                            embedding_matrix[idx] = vector * 1.1
                        elif vocab_word[0].isupper():
                            embedding_matrix[idx] = vector * 1.05
                        else:
                            embedding_matrix[idx] = vector
                
    except Exception as e:
        raise Exception(f"Error loading GloVe embeddings: {str(e)}")
    
    return embedding_matrix


# In[21]:


def load_glove_embeddings(word2idx: Dict[str, int], glove_path: str, case_sensitive: bool = False) -> torch.Tensor:
    embedding_matrix = torch.randn((len(word2idx), DatasetConfig.EMBEDDING_DIM)) * 0.1
    
    try:
        with gzip.open(glove_path, 'rt', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.FloatTensor([float(x) for x in values[1:]])
                
                if case_sensitive:
                    if word in word2idx:
                        embedding_matrix[word2idx[word]] = vector
                    
                    word_lower = word.lower()
                    for vocab_word in word2idx:
                        if vocab_word.lower() == word_lower:
                            idx = word2idx[vocab_word]
                            if vocab_word.isupper():
                                embedding_matrix[idx] = vector * 1.1
                            elif vocab_word[0].isupper():
                                embedding_matrix[idx] = vector * 1.05
                            else:
                                embedding_matrix[idx] = vector
                else:
                    word_lower = word.lower()
                    if word_lower in word2idx:
                        embedding_matrix[word2idx[word_lower]] = vector
                
    except Exception as e:
        raise Exception(f"Error loading GloVe embeddings: {str(e)}")
    
    return embedding_matrix


# In[22]:


# Save models
def predict_and_save(model, data_loader, output_path, original_sentences, device, with_chars=False):
    model.eval()
    sentence_idx = 0
    
    with torch.no_grad(), open(output_path, 'w') as f:
        for batch in data_loader:
            if with_chars:
                word_ids, char_ids, _, mask = batch
                word_ids = word_ids.to(device)
                char_ids = char_ids.to(device)
                logits = model(word_ids, char_ids)
            else:
                word_ids, _, mask = batch
                word_ids = word_ids.to(device)
                logits = model(word_ids)
            
            predictions = torch.argmax(logits, dim=-1)
            
            for pred, m in zip(predictions, mask):
                pred_tags = pred[m].cpu().numpy()
                orig_sentence = original_sentences[sentence_idx]
                
                for idx, (word, tag) in enumerate(zip(orig_sentence, pred_tags)):
                    f.write(f"{idx+1} {word} {DatasetConfig.ner_idx2tag[tag]}\n")
                f.write("\n")
                
                sentence_idx += 1


# In[33]:


def train_model(model, train_loader, dev_loader, optimizer, num_epochs, device, task_num, with_chars=False):
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    # Calculate class weights
    tag_counts = torch.zeros(len(DatasetConfig.ner_tag2idx))
    if with_chars:
        for _, _, tags, mask in train_loader:
            for tag, m in zip(tags, mask):
                for t in tag[m]:
                    tag_counts[t] += 1
    else:
        for _, tags, mask in train_loader:
            for tag, m in zip(tags, mask):
                for t in tag[m]:
                    tag_counts[t] += 1
    
    # Class weights calculation
    class_weights = (1.0 / (tag_counts + 1))
    class_weights = class_weights / class_weights.sum()
    class_weights[DatasetConfig.ner_tag2idx['O']] *= 0.8
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if with_chars:
                words, char_ids, tags, mask = batch
                words = words.to(device)
                char_ids = char_ids.to(device)
                tags = tags.to(device)
                mask = mask.to(device)
                outputs = model(words, char_ids)
            else:
                words, tags, mask = batch
                words = words.to(device)
                tags = tags.to(device)
                mask = mask.to(device)
                outputs = model(words)
            
            optimizer.zero_grad()
            
            # Compute loss only on masked positions
            active_logits = outputs.view(-1, outputs.size(-1))[mask.view(-1)]
            active_labels = tags.view(-1)[mask.view(-1)]
            
            loss = criterion(active_logits, active_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                if with_chars:
                    words, char_ids, tags, mask = batch
                    words = words.to(device)
                    char_ids = char_ids.to(device)
                    tags = tags.to(device)
                    mask = mask.to(device)
                    outputs = model(words, char_ids)
                else:
                    words, tags, mask = batch
                    words = words.to(device)
                    tags = tags.to(device)
                    mask = mask.to(device)
                    outputs = model(words)
                
                predictions = outputs.argmax(dim=2)
                
                for pred, label, m in zip(predictions, tags, mask):
                    pred_masked = pred[m].cpu().numpy()
                    label_masked = label[m].cpu().numpy()
                    all_predictions.extend(pred_masked)
                    all_labels.extend(label_masked)
        
        # Calculate metrics
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(PathConfig.OUTPUT_DIR, f'bilstm-{task_num}.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= DatasetConfig.PATIENCE:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break
    
    model.load_state_dict(best_model_state)
    return model


# In[25]:


def prepare_data(sentences: List[List[str]], labels: List[List[str]], 
                case_sensitive: bool = False, use_glove: bool = False) -> Tuple:
    """Prepare data loaders for training"""
    word2idx = build_vocab(sentences, case_sensitive)
    embedding_matrix = initialize_embeddings(
        word2idx, 
        PathConfig.GLOVE_100d_File if use_glove else None
    )
    
    # Create datasets
    train_dataset = NERDataset(train_sentences, train_labels, word2idx, case_sensitive)
    dev_dataset = NERDataset(dev_sentences, dev_labels, word2idx, case_sensitive)
    test_dataset = NERDataset(test_sentences, [['O'] * len(sent) for sent in test_sentences], 
                             word2idx, case_sensitive)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=DatasetConfig.BATCH_SIZE, 
                            shuffle=True, collate_fn=pad_sequence)
    dev_loader = DataLoader(dev_dataset, batch_size=DatasetConfig.BATCH_SIZE, 
                          shuffle=False, collate_fn=pad_sequence)
    test_loader = DataLoader(test_dataset, batch_size=DatasetConfig.BATCH_SIZE, 
                           shuffle=False, collate_fn=pad_sequence)
    
    return train_loader, dev_loader, test_loader, word2idx, embedding_matrix


# In[26]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_sentences, train_labels = load_datasets(DatasetConfig.train_data_path)
dev_sentences, dev_labels = load_datasets(DatasetConfig.dev_data_path)
test_sentences, _ = load_datasets(DatasetConfig.test_data_path)


# In[27]:


# Task 1: Basic BiLSTM (no GloVe)
print("\nTask 1: Training Basic BiLSTM...")
word2idx_task1 = build_vocab(train_sentences, case_sensitive=False)
embedding_matrix_task1 = torch.randn((len(word2idx_task1), DatasetConfig.EMBEDDING_DIM)) * 0.1

train_dataset_task1 = NERDataset(train_sentences, train_labels, word2idx_task1, case_sensitive=False)
dev_dataset_task1 = NERDataset(dev_sentences, dev_labels, word2idx_task1, case_sensitive=False)
test_dataset_task1 = NERDataset(test_sentences, [['O'] * len(sent) for sent in test_sentences], word2idx_task1, case_sensitive=False)

train_loader_task1 = DataLoader(train_dataset_task1, batch_size=DatasetConfig.BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
dev_loader_task1 = DataLoader(dev_dataset_task1, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence)
test_loader_task1 = DataLoader(test_dataset_task1, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence)

model1 = BLSTM_NER(len(word2idx_task1), len(DatasetConfig.ner_tag2idx), embedding_matrix_task1).to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=DatasetConfig.LEARNING_RATE, momentum=0.9, weight_decay=DatasetConfig.WEIGHT_DECAY, nesterov=True) 

train_model(model1, train_loader_task1, dev_loader_task1, optimizer1, DatasetConfig.NUM_EPOCHS, device, task_num=1)

predict_and_save(model1, dev_loader_task1, PathConfig.DEV_1_FILE_PATH, dev_sentences, device)
predict_and_save(model1, test_loader_task1, PathConfig.TEST_1_FILE_PATH, test_sentences, device)


# In[28]:


# get_ipython().system('python3 eval.py -p output/dev1.out -g data/dev')


# In[29]:


# !python3 eval.py -p output/test1.out -g data/test


# In[30]:


# Task 2: BiLSTM with GloVe and case-sensitivity
print("\nTask 2: Training BiLSTM with GloVe...")
word2idx_task2 = build_vocab(train_sentences, case_sensitive=True)
embedding_matrix_task2 = load_glove_embeddings(word2idx_task2, PathConfig.GLOVE_100d_File, case_sensitive=True)

train_dataset_task2 = NERDataset(train_sentences, train_labels, word2idx_task2, case_sensitive=True)
dev_dataset_task2 = NERDataset(dev_sentences, dev_labels, word2idx_task2, case_sensitive=True)
test_dataset_task2 = NERDataset(test_sentences, [['O'] * len(sent) for sent in test_sentences], word2idx_task2, case_sensitive=True)

train_loader_task2 = DataLoader(train_dataset_task2, batch_size=DatasetConfig.BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
dev_loader_task2 = DataLoader(dev_dataset_task2, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence)
test_loader_task2 = DataLoader(test_dataset_task2, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence)

model2 = BLSTM_NER(len(word2idx_task2), len(DatasetConfig.ner_tag2idx), embedding_matrix_task2).to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=DatasetConfig.LEARNING_RATE, momentum=0.9, weight_decay=DatasetConfig.WEIGHT_DECAY, nesterov=True)

train_model(model2, train_loader_task2, dev_loader_task2, optimizer2, DatasetConfig.NUM_EPOCHS, device, task_num=2)

predict_and_save(model2, dev_loader_task2, PathConfig.DEV_2_FILE_PATH, dev_sentences, device)
predict_and_save(model2, test_loader_task2, PathConfig.TEST_2_FILE_PATH, test_sentences, device)
    


# In[31]:


# get_ipython().system('python3 eval.py -p output/dev2.out -g data/dev')


# ## BONUS CNN 

# In[ ]:


# Bonus Task: BiLSTM with CNN for character information
print("\nBonus Task: Training BiLSTM with Character CNN...")
char2idx = build_char_vocab(train_sentences)

train_dataset_bonus = NERDatasetWithChars(train_sentences, train_labels, word2idx_task2, char2idx, case_sensitive=True)
dev_dataset_bonus = NERDatasetWithChars(dev_sentences, dev_labels, word2idx_task2, char2idx, case_sensitive=True)
test_dataset_bonus = NERDatasetWithChars(test_sentences, [['O'] * len(sent) for sent in test_sentences], word2idx_task2, char2idx, case_sensitive=True)

train_loader_bonus = DataLoader(train_dataset_bonus, batch_size=DatasetConfig.BATCH_SIZE, shuffle=True, collate_fn=pad_sequence_with_chars)
dev_loader_bonus = DataLoader(dev_dataset_bonus, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence_with_chars)
test_loader_bonus = DataLoader(test_dataset_bonus, batch_size=DatasetConfig.BATCH_SIZE, shuffle=False, collate_fn=pad_sequence_with_chars)

model_bonus = BLSTM_NER_with_Chars(len(word2idx_task2), len(DatasetConfig.ner_tag2idx), len(char2idx), embedding_matrix_task2).to(device)
optimizer_bonus = optim.SGD(model_bonus.parameters(), lr=0.005, momentum=0.9)

train_model(model_bonus, train_loader_bonus, dev_loader_bonus, optimizer_bonus, DatasetConfig.NUM_EPOCHS, device, task_num=3, with_chars=True)

predict_and_save(model_bonus, dev_loader_bonus, PathConfig.DEV_BONUS_FILE_PATH, dev_sentences, device, with_chars=True)
predict_and_save(model_bonus, test_loader_bonus, PathConfig.TEST_BONUS_FILE_PATH, test_sentences, device, with_chars=True)


# In[ ]:


# get_ipython().system('python3 eval.py -p output/dev-bonus.out -g data/dev')


# ## REFERENCES
# - https://huggingface.co/docs/datasets/load_hub
# - https://nlp.stanford.edu/projects/glove/
# - https://github.com/tqdm/tqdm
# - https://github.com/guillaumegenthial/sequence_tagging
# - https://github.com/XuezheMax/LasagneNLP
# - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# 

# In[ ]:




