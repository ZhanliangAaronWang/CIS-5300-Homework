from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *



""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n

from tagger_utils import *
import numpy as np
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import numpy as np
from multiprocessing import Pool
import time
from tagger_utils import *
from sklearn.neural_network import MLPClassifier
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary."""
        self.data = None
        self.all_tags = None
        self.tag2idx = None
        self.idx2tag = None
        self.word2idx = None
        self.idx2word = None
        self.unigrams = None
        self.bigrams = None
        self.trigrams = None
        self.emissions = None
        self.suffix2pos = None  

        self.mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)

        # Hyperparameters from the configuration
        if SMOOTHING == LAPLACE:
            self.smoothing_factor = LAPLACE_FACTOR 
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is not None:
                self.smoothing_factor = LAMBDAS 
            else:
                self.smoothing_factor = 0.1  
        else:
            self.smoothing_factor = 0  
        self.unk_token = '<UNK>'
        self.wordnet_pos_mapping = {
            'n': 'NOUN',
            'v': 'VERB',
            'a': 'ADJ',
            'r': 'ADV'
        }
            
    def preprocess(self, sentence):
        """Preprocesses the sentence by converting to lowercase if CAPITALIZATION is set."""
        if not CAPITALIZATION:
            return [word.lower() for word in sentence]
        return sentence

    def get_unigrams(self):
        """Computes unigrams (tag probabilities)."""
        unigrams = np.zeros(len(self.all_tags))
        for sentence in self.data[1]:
            for tag in sentence:
                unigrams[self.tag2idx[tag]] += 1
        return (unigrams + EPSILON) / (np.sum(unigrams) + EPSILON)

    def get_bigrams(self):
        """Computes bigrams (transition probabilities)."""
        bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        for sentence in self.data[1]:
            for i in range(len(sentence) - 1):
                bigrams[self.tag2idx[sentence[i]], self.tag2idx[sentence[i + 1]]] += 1
        return (bigrams + self.smoothing_factor) / (np.sum(bigrams, axis=1, keepdims=True) + self.smoothing_factor * len(self.all_tags))

    def get_trigrams(self):
        """Computes trigrams (transition probabilities)."""
        trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        for sentence in self.data[1]:
            for i in range(len(sentence) - 2):
                trigrams[self.tag2idx[sentence[i]], self.tag2idx[sentence[i + 1]], self.tag2idx[sentence[i + 2]]] += 1
        return (trigrams + self.smoothing_factor) / (np.sum(trigrams, axis=2, keepdims=True) + self.smoothing_factor * len(self.all_tags))

    def get_emissions(self):
        """Computes emission probabilities."""
        emissions = np.zeros((len(self.all_tags), len(self.word2idx)))
        for sentence, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(sentence, tags):
                emissions[self.tag2idx[tag], self.word2idx.get(word, self.word2idx[self.unk_token])] += 1
        return (emissions + self.smoothing_factor) / (np.sum(emissions, axis=1, keepdims=True) + self.smoothing_factor * len(self.word2idx))

    def build_suffix_pos_mapping(self):
        """Builds a mapping from suffixes to POS tag distributions."""
        self.suffix2pos = {}
        for sentence, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(sentence, tags):
                suffix = word[-3:] if len(word) > 3 else word
                if suffix not in self.suffix2pos:
                    self.suffix2pos[suffix] = np.zeros(len(self.all_tags))
                self.suffix2pos[suffix][self.tag2idx[tag]] += 1
        for suffix in self.suffix2pos:
            self.suffix2pos[suffix] = (self.suffix2pos[suffix] + EPSILON) / (np.sum(self.suffix2pos[suffix]) + EPSILON)

    def build_mlp_features(self, word):
        """Generates features for the MLP model."""
        features = [
            len(word),
            int(word[0].isupper()), 
            int(any(char.isdigit() for char in word)),
            self.get_suffix_index(word), 
        ]
        return features

    def get_suffix_index(self, word):
        """Gets the index of the suffix."""
        suffix = word[-3:] if len(word) > 3 else word
        return hash(suffix) % 1000  

    def train_discriminative_model(self):
        """Trains the MLP model using extracted features."""
        features = []
        labels = []
        for sentence, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(sentence, tags):
                features.append(self.build_mlp_features(word))
                labels.append(self.tag2idx[tag])
        self.mlp.fit(features, labels)

    def predict_pos_with_mlp(self, word):
        """Predicts the POS tag for a word using the MLP model."""
        features = self.build_mlp_features(word)
        probabilities = self.mlp.predict_proba([features])[0]
        return probabilities 

    def train(self, data):
        """Trains the model by computing transition and emission probabilities."""
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Create the vocabulary and add <UNK> for unknown words
        all_words = set([w for sentence in data[0] for w in sentence])
        all_words.add(self.unk_token)
        self.word2idx = {word: idx for idx, word in enumerate(all_words)}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.unigrams = self.get_unigrams()
        self.bigrams = self.get_bigrams()
        self.trigrams = self.get_trigrams()
        self.emissions = self.get_emissions()
        self.build_suffix_pos_mapping() 
        self.train_discriminative_model() 

    def get_wordnet_pos(self, word):
        """Gets the POS tag from WordNet if available."""
        synsets = wn.synsets(word)
        if synsets:
            wordnet_pos = synsets[0].pos()
            return self.wordnet_pos_mapping.get(wordnet_pos, None)
        return None

    def get_suffix_pos_probability(self, word):
        """Gets the POS tag distribution based on the word's suffix."""
        suffix = word[-3:] if len(word) > 3 else word
        if suffix in self.suffix2pos:
            return self.suffix2pos[suffix]
        else:
            return np.ones(len(self.all_tags)) / len(self.all_tags)

    def handle_unknown_word(self, word):
        """Handles unknown words by combining suffix-based probabilities and MLP predictions."""
        suffix_pos_prob = self.get_suffix_pos_probability(word)
        mlp_pos_prob = self.predict_pos_with_mlp(word)

        combined_prob = (suffix_pos_prob + mlp_pos_prob) / 2
        return combined_prob  

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""
        prob = 1.0
        for i in range(len(sequence)):
            word = sequence[i]
            tag = tags[i]
            tag_idx = self.tag2idx[tag]
            word_idx = self.word2idx.get(word, self.word2idx[self.unk_token])
            if i == 0:
                prob *= self.unigrams[tag_idx]
            elif i == 1:
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                prob *= self.bigrams[prev_tag_idx, tag_idx]
            else:
                prev_prev_tag_idx = self.tag2idx[tags[i - 2]]
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                prob *= self.trigrams[prev_prev_tag_idx, prev_tag_idx, tag_idx]
            if word in self.word2idx:
                prob *= self.emissions[tag_idx, word_idx]
            else:
                unk_emission_prob = self.handle_unknown_word(word)
                prob *= unk_emission_prob[tag_idx]
        return prob

    def inference(self, sequence):
        """Tags a sequence with part of speech tags."""
        if INFERENCE == GREEDY:
            return self.greedy_decode(sequence)
        elif INFERENCE == BEAM:
            return self.beam_search(sequence, k=BEAM_K)
        elif INFERENCE == VITERBI:
            return self.viterbi_decode(sequence)
        return None

    def viterbi_decode(self, sequence):
        """Performs Viterbi decoding."""
        sequence = self.preprocess(sequence)
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for tag in self.all_tags:
            tag_idx = self.tag2idx[tag]
            word = sequence[0]
            if word in self.word2idx:
                emission_prob = self.emissions[tag_idx, self.word2idx[word]]
            else:
                emission_prob = self.handle_unknown_word(word)[tag_idx]
            V[0][tag] = self.unigrams[tag_idx] * emission_prob
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sequence)):
            V.append({})
            newpath = {}
            word = sequence[t]
            if word in self.word2idx:
                emission_probs = self.emissions[:, self.word2idx[word]]
            else:
                emission_probs = self.handle_unknown_word(word)
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag]
                (prob, prev_tag) = max(
                    (V[t - 1][ptag] * self.bigrams[self.tag2idx[ptag], tag_idx] * emission_probs[tag_idx], ptag)
                    for ptag in self.all_tags
                )
                V[t][tag] = prob
                newpath[tag] = path[prev_tag] + [tag]
            path = newpath

        # Find the final most probable tag sequence
        n = len(sequence) - 1
        (prob, state) = max((V[n][tag], tag) for tag in self.all_tags)
        return path[state]

    def greedy_decode(self, sequence):
        """Performs greedy decoding."""
        sequence = self.preprocess(sequence)
        tags = []
        for word in sequence:
            if word in self.word2idx:
                emission_probs = self.emissions[:, self.word2idx[word]]
            else:
                emission_probs = self.handle_unknown_word(word)
            best_tag_idx = np.argmax(emission_probs)
            best_tag = self.idx2tag[best_tag_idx]
            tags.append(best_tag)
        return tags

    def beam_search(self, sequence, k=3):
        """Performs beam search decoding."""
        pass




if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    INFERENCE = VITERBI
    SMOOTHING = LAPLACE 
    CAPITALIZATION = False  
    TNT_UNK = True  
    pos_tagger.train(train_data)

    # Use the evaluate function to evaluate the model
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))

    # TODO: Save predictions to file to update the leaderboard
