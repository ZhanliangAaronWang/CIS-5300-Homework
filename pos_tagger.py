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


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
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
        
        # Hyperparameters
        self.smoothing_factor = 1e-5  # for add-k smoothing
        self.unk_token = '<UNK>'
    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        unigrams = np.zeros(len(self.all_tags))
        for sentence in self.data[1]:
            for tag in sentence:
                unigrams[self.tag2idx[tag]] += 1
        return unigrams / np.sum(unigrams)

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        for sentence in self.data[1]:
            for i in range(len(sentence) - 1):
                bigrams[self.tag2idx[sentence[i]], self.tag2idx[sentence[i+1]]] += 1
        return (bigrams + self.smoothing_factor) / (np.sum(bigrams, axis=1, keepdims=True) + self.smoothing_factor * len(self.all_tags))

    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        for sentence in self.data[1]:
            for i in range(len(sentence) - 2):
                trigrams[self.tag2idx[sentence[i]], self.tag2idx[sentence[i+1]], self.tag2idx[sentence[i+2]]] += 1
        return (trigrams + self.smoothing_factor) / (np.sum(trigrams, axis=2, keepdims=True) + self.smoothing_factor * len(self.all_tags))

    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        emissions = np.zeros((len(self.all_tags), len(self.word2idx)))
        for sentence, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(sentence, tags):
                emissions[self.tag2idx[tag], self.word2idx[word]] += 1
        return (emissions + self.smoothing_factor) / (np.sum(emissions, axis=1, keepdims=True) + self.smoothing_factor * len(self.word2idx)) 
    

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}
        
        all_words = set([w for sentence in data[0] for w in sentence])
        all_words.add(self.unk_token)
        self.word2idx = {word: idx for idx, word in enumerate(all_words)}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        
        self.unigrams = self.get_unigrams()
        self.bigrams = self.get_bigrams()
        self.trigrams = self.get_trigrams()
        self.emissions = self.get_emissions()

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        prob = 1.0
        for i in range(len(sequence)):
            if i == 0:
                prob *= self.unigrams[self.tag2idx[tags[i]]]
            elif i == 1:
                prob *= self.bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]]
            else:
                prob *= self.trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]]
            prob *= self.emissions[self.tag2idx[tags[i]], self.word2idx.get(sequence[i], self.word2idx[self.unk_token])]
        return prob

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        ## TODO
        return []
    
    def greedy_decode(self, sequence):
        """Performs greedy decoding."""
        tags = []
        for word in sequence:
            best_tag = max(self.all_tags, key=lambda tag: self.emissions[self.tag2idx[tag], self.word2idx.get(word, self.word2idx[self.unk_token])])
            tags.append(best_tag)
        return tags

    def viterbi_decode(self, sequence):
        """Performs Viterbi decoding."""
        V = [{}]
        path = {}
        
        # Initialize base cases (t == 0)
        for tag in self.all_tags:
            V[0][tag] = self.unigrams[self.tag2idx[tag]] * self.emissions[self.tag2idx[tag], self.word2idx.get(sequence[0], self.word2idx[self.unk_token])]
            path[tag] = [tag]
        
        # Run Viterbi for t > 0
        for t in range(1, len(sequence)):
            V.append({})
            newpath = {}
            for tag in self.all_tags:
                (prob, state) = max((V[t-1][prev_tag] * self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]] * 
                                     self.emissions[self.tag2idx[tag], self.word2idx.get(sequence[t], self.word2idx[self.unk_token])], prev_tag) 
                                    for prev_tag in self.all_tags)
                V[t][tag] = prob
                newpath[tag] = path[state] + [tag]
            path = newpath
        
        (prob, state) = max((V[len(sequence) - 1][tag], tag) for tag in self.all_tags)
        return path[state]

if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO
