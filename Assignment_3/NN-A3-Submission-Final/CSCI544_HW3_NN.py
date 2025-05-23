import numpy as np
import pandas as pd
import itertools
import json
import os
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from typing import List

class PathConfig:
    HW3_DIR = os.getcwd() # os.path.dirname(os.getcwd())
    OUTPUT_DIR = os.path.join(HW3_DIR, "output")
    DATASET_PATH = os.path.join(HW3_DIR, "data")
    VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, "vocab.txt")
    HMM_MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, "hmm.json")
    GREEDY_EVAL_OUPUT_LOC = os.path.join(OUTPUT_DIR, "greedy.out")
    VITERBI_EVAL_OUPUT_LOC = os.path.join(OUTPUT_DIR, "viterbi.out")
    EVAL_FILE = os.path.join(HW3_DIR, "eval.py")

class DatasetConfig:
    cols = ["index", "sentences", "labels"]
    train_data_path = os.path.join(PathConfig.DATASET_PATH, "train")
    dev_data_path = os.path.join(PathConfig.DATASET_PATH, "dev")
    test_data_path = os.path.join(PathConfig.DATASET_PATH, "test")


class VocabGenConfig:
    UNK_TOKEN = "<unk>"
    THRESHOLD = 2
    FILE_HEADER = ["word", "index", "frequency"]
    VOCAB_FILE = PathConfig.VOCAB_FILE_PATH


class HMMConfig:
    HMM_MODEL_SAVED = PathConfig.HMM_MODEL_FILE_PATH

class Dataset:
    def __init__(self, path, split="train"):
        self.path = path
        self.split = split
        self.data: pd.DataFrame = None
        

    # Reads tsv file, groups words into sentences, handle missing POS tags
    def _read_data(self):
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
    
        with open(self.path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split("\t")
    
                    if len(parts) == 3:  
                        _, word, pos = parts
                        current_sentence.append(word)
                        current_labels.append(pos)  
    
                    elif len(parts) == 2:  
                        _, word = parts
                        current_sentence.append(word)
                        current_labels.append(None)  
                    
                else:  
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
    
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
        self.data = pd.DataFrame({"sentence": sentences, "labels": labels})
        return self.data



    def prepare_dataset(self):
        self._read_data()
        return self.data

    
    # Returns sentences with its POS tags as lists of tuples.
    def get_sent_pos_tags(self):
        return [
            list(zip(sentence, labels))
            for sentence, labels in zip(self.data["sentence"], self.data["labels"])
        ]


# Prep Train dataset
train_dataset = Dataset(path=DatasetConfig.train_data_path)
df_train = train_dataset.prepare_dataset()
# print(df_train.shape)
# df_train.head()

# Prep Dev dataset
dev_dataset = Dataset(path=DatasetConfig.dev_data_path)
df_valid = dev_dataset.prepare_dataset()
# print(df_valid.shape)
# df_valid.head()

# for creating future copies
raw_test_df = Dataset(path=DatasetConfig.test_data_path)._read_data()

test_dataset = Dataset(path=DatasetConfig.test_data_path)
df_test = test_dataset.prepare_dataset()
# print(df_test.shape)
# df_test.head()


# TASK 1: Vocabulary Generator
class VocabularyGenerator:
    def __init__(
        self, threshold: int, unk_token: str = None, save: bool = False, path: str = None):
        self.threshold = threshold
        self.unk_token = unk_token if unk_token is not None else VocabGenConfig.UNK_TOKEN
        self._save = save
        self.path = path if path is not None else VocabGenConfig.VOCAB_FILE

    
    def word_frequency(self, data, sent_col_name):
        word_freq = (
            data[sent_col_name]
            .explode()
            .value_counts()
            .rename_axis("word")
            .reset_index(name="frequency")
        )
        return word_freq

    
    def generate_vocabulary(self, data: pd.DataFrame, sent_col_name: str):
        word_frequency_df = self.word_frequency(data, sent_col_name)
        word_frequency_df["word"] = word_frequency_df.apply(
            lambda row: self.unk_token if row["frequency"] < self.threshold else row["word"],
            axis=1,
        )
        word_frequency_df = word_frequency_df.groupby("word", as_index=False)["frequency"].sum()
        
        if self.unk_token not in word_frequency_df["word"].values:
            word_frequency_df = pd.concat([
                pd.DataFrame([[self.unk_token, 0]], columns=["word", "frequency"]),
                word_frequency_df
            ], ignore_index=True)

        # descending count
        word_frequency_df = word_frequency_df.sort_values(by="frequency", ascending=False)
        word_frequency_df["index"] = range(len(word_frequency_df))
        self.vocab = word_frequency_df 
        
        if self._save:
            self.save_vocab(word_frequency_df, self.path)
        return word_frequency_df
        

    def save_vocab(self, word_frequency_df, path):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            for _, row in word_frequency_df.iterrows():
                file.write(f"{row['word']}\t{row['index']}\t{row['frequency']}\n")

    
    def replace_unk_words(self, data: pd.DataFrame, sent_col_name: str):
        if not hasattr(self, "vocab"):  
            raise ValueError("Vocabulary not generated. Run `generate_vocabulary()` first.")
        vocab_set = set(self.vocab["word"])  
        
        def replace_word(word):
            return word if word in vocab_set else self.unk_token
            
        data[sent_col_name] = data[sent_col_name].apply(
            lambda sentence: [replace_word(word) for word in sentence]
        )
        return data


df_train_exploded = df_train["labels"].explode()
my_vocab_generator = VocabularyGenerator(
    threshold=VocabGenConfig.THRESHOLD, unk_token=VocabGenConfig.UNK_TOKEN, save=True
)
vocab_df = my_vocab_generator.generate_vocabulary(df_train, "sentence")
vocab_df.head(10)

print("Selected threshold value for unknown words: ", VocabGenConfig.THRESHOLD)
print("Vocabulary size: ", vocab_df.shape[0])
print(
    "Total count of special tokens <unk>: ",
    int(vocab_df[vocab_df["word"] == "<unk>"].frequency),
)

unique_pos_tags = df_train["labels"].explode().dropna().unique()
# print("Number of unique POS tags =", unique_pos_tags.shape[0])
# print("Unique Part-of-speech tags:\n", unique_pos_tags)
unique_pos_tags = unique_pos_tags.tolist()

train_sent_with_pos_tags = train_dataset.get_sent_pos_tags()
# train_sent_with_pos_tags[0]

dev_sent_with_pos_tags = dev_dataset.get_sent_pos_tags()
# dev_sent_with_pos_tags[0]

test_sent_with_pos_tags = test_dataset.get_sent_pos_tags()
# test_sent_with_pos_tags[0]

class HMModel:
    def __init__(self, vocab_file_path: str, labels: List[str]):
        self.vocab = self.read_vocab_content(vocab_file_path)
        self.labels = labels
        # HMM Parameters
        self.states = list(self.labels)
        self.priors = None
        self.transitions = None
        self.emissions = None
        # Laplace Smoothing
        self.smoothing_constant = 1e-10

    
    def read_vocab_content(self, vocab_file_path: str):
        vocab = pd.read_csv(vocab_file_path, sep="\t", names=VocabGenConfig.FILE_HEADER)
        if VocabGenConfig.UNK_TOKEN not in vocab["word"].values:
            unk_row = pd.DataFrame([[VocabGenConfig.UNK_TOKEN, 0, 0]], columns=vocab.columns)
            vocab = pd.concat([unk_row, vocab], ignore_index=True)
        return vocab

        
    def initialize_model_params(self):
        self.states = list(self.labels)
        num_states = len(self.labels)
        num_obs = len(self.vocab)

        # probability matrices with 0
        self.priors = np.zeros(num_states)
        self.emissions = np.zeros((num_states, num_obs))
        self.transitions = np.zeros((num_states, num_states))


    # Handle zero probabilities using Laplace smoothing
    def smoothen_probab(self, prob_mat: np.array):
        prob_mat = np.where(prob_mat == 0, self.smoothing_constant, prob_mat)
        return np.nan_to_num(prob_mat, nan=0.0)  


    # Compute initial state probabilities
    def compute_prior(self, train_data):
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}
        num_sentences = len(train_data)
        if num_sentences == 0:
            self.priors = np.full(len(self.labels), 1 / len(self.labels))  
            return
        for sentence in train_data:
            label = sentence[0][1]
            state_idx = tag_to_index[label]
            self.priors[state_idx] += 1
        self.priors /= num_sentences
        self.priors = self.smoothen_probab(self.priors)


    # Compute transition probabilities between states
    def compute_transition(self, train_data):
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}
        for sentence in train_data:
            label_indices = [tag_to_index.get(label) for _, label in sentence]
            for i in range(1, len(label_indices)):
                prev_state = label_indices[i - 1]
                curr_state = label_indices[i]
                self.transitions[prev_state, curr_state] += 1
        row_agg = self.transitions.sum(axis=1)[:, np.newaxis]
        # Prevent division by zero
        row_agg[row_agg == 0] = 1  
        self.transitions = self.transitions / row_agg
        self.transitions = self.smoothen_probab(self.transitions)


    # Compute emission probabilities
    def compute_emission(self, train_data):
        word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}
        for sentence in train_data:
            for word, label in sentence:
                state_idx = tag_to_index[label]
                word_idx = word_to_index.get(word, word_to_index[VocabGenConfig.UNK_TOKEN])
                self.emissions[state_idx, word_idx] += 1
        row_agg = self.emissions.sum(axis=1)[:, np.newaxis]
        row_agg[row_agg == 0] = 1  
        self.emissions = self.emissions / row_agg
        self.emissions = self.smoothen_probab(self.emissions)


    # Train HMM
    def fit(self, train_data: pd.DataFrame):
        self.initialize_model_params()
        self.compute_prior(train_data)
        self.compute_transition(train_data)
        self.compute_emission(train_data)


        
    @property
    def get_all_probab_matrices(self):
        return self.priors, self.transitions, self.emissions

    
    def save_model(self, file_path=None):
        if file_path is None:
            file_path = HMMConfig.HMM_MODEL_SAVED
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        state_index = {s: i for i, s in enumerate(self.states)}
        # Store transition probabilities
        transition_prob = {
            f"({s1}, {s2})": self.transitions[state_index[s1], state_index[s2]]
            for s1, s2 in itertools.product(self.states, repeat=2)
        }
        # Store emission probabilities
        emission_prob = {
            f"({s}, {w})": p
            for s in self.states
            for w, p in zip(self.vocab["word"], self.emissions[state_index[s], :])
        }
        model_params = {"transition": transition_prob, "emission": emission_prob}
        # save content to file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(model_params, json_file, indent=4)


model = HMModel(vocab_file_path=VocabGenConfig.VOCAB_FILE, labels=unique_pos_tags)
if not train_sent_with_pos_tags or all(lbl is None for _, lbl in itertools.chain(*train_sent_with_pos_tags)):
    raise ValueError("Training data is empty or contains invalid pos_labels.")

model.fit(train_sent_with_pos_tags)

p, t, e = model.get_all_probab_matrices
print("Number of Priors =", len(p.flatten()))
print("Number of Transition Parameters =", len(t.flatten()))
print("Number of Emission Parameters =", len(e.flatten()))

model.save_model()

class GreedyDecoding:

    # Initialize Greedy Decoding.
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab

        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))

        self.unk_token_idx = self.word_to_index.get(VocabGenConfig.UNK_TOKEN, 0)

        # get initial scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

        # Laplace smoothing constant
        self.smoothing_constant = 1e-10


    # Decode a single sentence using Greedy Decoding 
    def decode_single_sent(self, sentence):
        predicted_tags = []
        prev_tag_idx = None

        for word in sentence:
            word_idx = self.word_to_index.get(word, self.unk_token_idx)

            if prev_tag_idx is None:
                scores = np.maximum(self.priors_emissions[:, word_idx], self.smoothing_constant)
            else:
                scores = np.maximum(self.transitions[prev_tag_idx] * self.emissions[:, word_idx], self.smoothing_constant)

            prev_tag_idx = np.argmax(scores)
            predicted_tags.append(self.states[prev_tag_idx])
        return predicted_tags


    # Perform Greedy Decoding on multiple sentences
    def decode(self, sentences):
        return [self.decode_single_sent([word for word, _ in sentence]) for sentence in sentences]


def calc_accuracy(redicted_seq, true_seq):
    total = 0
    correct = 0

    for true_label, predicted_label in zip(true_seq, redicted_seq):
        for true_tag, predicted_tag in zip(true_label, predicted_label):
            total += 1
            if true_tag == predicted_tag:
                correct += 1
    gr_accuracy = correct / total
    return gr_accuracy

# Apply Greedy Decoding on development data
greedy_decoder = GreedyDecoding(p, t, e, model.states, model.vocab)
predicted_dev_tags = greedy_decoder.decode(dev_sent_with_pos_tags)

gr_acc = calc_accuracy(predicted_dev_tags, df_valid.labels.tolist())
print("Greedy Decoding Algo Accuracy: ", round(gr_acc, 4))

# Apply Greedy Decoding on Test data
predicted_test_tags = greedy_decoder.decode(test_sent_with_pos_tags)
df_greedy_preds = raw_test_df.copy(deep=True)
df_greedy_preds["labels"] = predicted_test_tags
df_greedy_preds.head()

with open(PathConfig.GREEDY_EVAL_OUPUT_LOC, "w", encoding="utf-8") as f:
    word_index = 1  
    for sentence, predicted_tags in zip(test_sent_with_pos_tags, predicted_test_tags):
        for word, tag in zip(sentence, predicted_tags):
            word = word[0] if isinstance(word, tuple) else word  
            f.write(f"{word_index}\t{word}\t{tag}\n")  
            word_index += 1
        f.write("\n")  


# # For dev file eval check
# with open(PathConfig.GREEDY_EVAL_OUPUT_LOC, "w", encoding="utf-8") as f:
#     word_index = 1  
#     for sentence, predicted_tags in zip(dev_sent_with_pos_tags, predicted_dev_tags):
#         for word, tag in zip(sentence, predicted_tags):
#             word = word[0] if isinstance(word, tuple) else word  
#             f.write(f"{word_index}\t{word}\t{tag}\n")  
#             word_index += 1
#         f.write("\n")  


# import subprocess
# gold_file = DatasetConfig.test_data_path # test
# pred_file = PathConfig.GREEDY_EVAL_OUPUT_LOC

# print()
# # Run eval.py to compute accuracy
# subprocess.run(["python3", PathConfig.EVAL_FILE, "-p", pred_file, "-g", gold_file])


# REFERENCE : https://www.nltk.org/book/ch08.html 
# https://digitalscholarship.unlv.edu/cgi/viewcontent.cgi?article=2008&context=thesesdissertations


class ViterbiAlgoDecoding:
    # same as greedy decoding
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab
        self.num_states = len(self.states)

        # Index Conversion dictionary for mapping
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_idx = dict(zip(self.vocab["word"], self.vocab["index"]))

        # Precompute scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

    
    def init_variables(self, sentence):
        Vb = np.zeros((len(sentence), self.num_states))
        path = np.zeros((len(sentence), self.num_states), dtype=int)
        word_idx = np.array([
            self.word_to_idx.get(word, self.word_to_idx.get(VocabGenConfig.UNK_TOKEN, 0)) 
            for word in sentence
        ])
        return Vb, path, word_idx

    
    def decode_single_sent(self, sentence):
        V, path, word_idx = self.init_variables(sentence)

        self.transitions = np.where(self.transitions == 0, 1e-10, self.transitions)
        self.emissions = np.where(self.emissions == 0, 1e-10, self.emissions)
        V[0] = np.log(np.where(self.priors_emissions[:, word_idx[0]] == 0, 1e-10, self.priors_emissions[:, word_idx[0]]))


        for t in range(1, len(sentence)):
            # Compute scores
            scores = (
                V[t - 1, :, np.newaxis]
                + np.log(self.transitions)
                + np.log(self.emissions[:, word_idx[t]])
            )
            V[t] = np.max(scores, axis=0)
            path[t] = np.argmax(scores, axis=0)

        # Backtracking
        predicted_tags = [0] * len(sentence)
        predicted_tags[-1] = np.argmax(V[-1])
        for t in range(len(sentence) - 1, 0, -1): 
            predicted_tags[t - 1] = path[t, predicted_tags[t]]
        predicted_tags = [self.states[tag_idx] for tag_idx in predicted_tags]
        return predicted_tags


    
    def decode(self, sentences):
        predicted_tags_list = []
        for sentence in sentences:
            words = [word[0] if isinstance(word, tuple) else word for word in sentence]
            predicted_tags = self.decode_single_sent(words)
            predicted_tags_list.append(predicted_tags)
        return predicted_tags_list

        

# Apply Greedy Decoding on development data
viterbi_decoder = ViterbiAlgoDecoding(p, t, e, model.states, model.vocab)
predicted_dev_tags_viterbi = viterbi_decoder.decode(dev_sent_with_pos_tags)

acc_v = calc_accuracy(predicted_dev_tags_viterbi, df_valid.labels.tolist())
print("Viterbi Decoding Algo Accuracy: ", round(acc_v, 4))

predicted_test_tags_v = viterbi_decoder.decode(test_sent_with_pos_tags)

df_viterbi_preds = raw_test_df.copy(deep=True)
df_viterbi_preds["labels"] = pd.Series(predicted_test_tags_v).reset_index(drop=True)

# df_viterbi_preds.head(10)

with open(PathConfig.VITERBI_EVAL_OUPUT_LOC, "w", encoding="utf-8") as f:
    word_index = 1 
    for sentence, predicted_tags in zip(test_sent_with_pos_tags, predicted_test_tags_v):
        for word, tag in zip(sentence, predicted_tags):
            word = word[0] if isinstance(word, tuple) else word  
            f.write(f"{word_index}\t{word}\t{tag}\n")  
            word_index += 1
        f.write("\n") 

# for dev file against eval
# with open(PathConfig.VITERBI_EVAL_OUPUT_LOC, "w", encoding="utf-8") as f:
#     word_index = 1 
#     for sentence, predicted_tags in zip(dev_sent_with_pos_tags, predicted_dev_tags_viterbi):
#         for word, tag in zip(sentence, predicted_tags):
#             word = word[0] if isinstance(word, tuple) else word  
#             f.write(f"{word_index}\t{word}\t{tag}\n")  
#             word_index += 1
#         f.write("\n")


# import subprocess

# gold_file = DatasetConfig.test_data_path # test
# pred_file = PathConfig.VITERBI_EVAL_OUPUT_LOC

# print()
# # Run eval.py to compute accuracy
# subprocess.run(["python3", PathConfig.EVAL_FILE, "-p", pred_file, "-g", gold_file])


# %%bash
# cp -r ./output/* ../A3-FILES/output-files/.


