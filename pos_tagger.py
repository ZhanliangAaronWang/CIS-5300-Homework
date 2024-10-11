from multiprocessing import Pool
from tagger_constants import *
import numpy as np
import time
import string
from tagger_utils import *
from itertools import tee
from pos_tagger_unknown import POSTagger_MLP
import csv
import string
from typing import List, Tuple, Dict
from itertools import tee
import numpy as np
""" Contains the part of speech tagger class. """

def flatten_data(data_words, data_tags):
    flattened_words = [word for sentence in data_words for word in sentence]
    flattened_tags = [tag for tag_list in data_tags for tag in tag_list]
    return flattened_words, flattened_tags

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def contains_dash(inputString):
    return '-' in inputString

def is_punctuation(word):
    return word in string.punctuation

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
    processes = 8
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
    #adding my manual code
    misclassification_count = 0
    for i in range(n):
        for j in range(len(sentences[i])):
            # Check if the word is unknown and misclassified
            if sentences[i][j] not in model.word2idx.keys() and tags[i][j] != predictions[i][j]:
                print(f"Sentence: {i}, Word: {sentences[i][j]}, Actual Tag: {tags[i][j]}, Predicted Tag: {predictions[i][j]}")
                misclassification_count += 1
                # Stop after printing 10 misclassifications
                if misclassification_count >= 10:
                    break
        if misclassification_count >= 10:
            break
    print("now look at words that are known but still predicted incorrecly.")
    misclassification_count = 0 
    for i in range(n):
        for j in range(len(sentences[i])):
            # Check if the word is unknown and misclassified
            if sentences[i][j] in model.word2idx.keys() and tags[i][j] != predictions[i][j]:
                print(f"Sentence: {i}, Word: {sentences[i][j]}, Actual Tag: {tags[i][j]}, Predicted Tag: {predictions[i][j]}")
                misclassification_count += 1
                # Stop after printing 10 misclassifications
                if misclassification_count >= 10:
                    break
        if misclassification_count >= 10:
            break
    #end of manual code
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


class POSTagger:
    def __init__(self, inference_method: str, smoothing_method: str = None) -> None:
        """Initializes the tagger model parameters and anything else necessary."""
        self.smoothing_method: str = smoothing_method
        self.inference_method: str = inference_method
        self.unigrams: np.ndarray = None
        self.bigrams: np.ndarray = None
        self.trigrams: np.ndarray = None
        self.lexical: np.ndarray = None
        self.ngram: int = None
        self.num_words: int = -1
        self.punct: set = set(string.punctuation)
        self.noun_suffix: List[str] = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
        self.verb_suffix: List[str] = ["ate", "ify", "ise", "ize"]
        self.adj_suffix: List[str] = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
        self.adv_suffix: List[str] = ["ward", "wards", "wise"]

    def get_unigrams(self) -> None:
        """Computes unigrams."""
        self.unigrams = np.array([sum(x.count(tag) for x in self.data_tags) / self.num_words for tag in self.all_tags])

    def get_unigrams_words(self) -> None:
        """Computes unigrams for words."""
        self.unigrams_words = np.array([sum(x.count(word) for x in self.data_words) / self.num_words for word in self.all_words])

    def get_bigrams(self) -> None:
        """Computes bigrams."""
        if self.unigrams is None:
            self.get_unigrams()
        bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b) 
        
        for document in self.data_tags:
            for curr, next_word in pairwise(document):
                if self.smoothing_method == LAPLACE:
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (LAPLACE_FACTOR + self.unigrams[self.tag2idx[curr]] * self.num_words)
                else: 
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (self.unigrams[self.tag2idx[curr]] * self.num_words)
        
        if self.smoothing_method == LAPLACE:
            for i in range(len(bigrams)):
                num_log = np.log(LAPLACE_FACTOR) - np.log(len(self.all_tags))
                denom_log = np.log(self.unigrams[i] * self.num_words + LAPLACE_FACTOR)
                bigrams[i] += np.exp(num_log - denom_log)
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2 = BIGRAM_LAMBDAS
            for i in range(len(bigrams)):
                for j in range(len(bigrams[i])):
                    bigrams[i,j] = lambda_1 * bigrams[i,j] + lambda_2 * self.unigrams[j]
        
        self.bigrams = bigrams
    
    def get_trigrams(self) -> None:
        """Computes trigrams."""
        if self.bigrams is None:
            self.get_bigrams()
        trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        bigram_denominators = np.zeros((len(self.all_tags), len(self.all_tags)))
        
        def triplewise(iterable):
            """s -> (s0,s1,s2), (s1,s2,s3), (s2, s3, s4), ..."""
            a, b, c = tee(iterable, 3)
            next(b, None)
            next(c, None)
            next(c, None)
            return zip(a, b, c) 
        
        for document in self.data_tags:
            for curr, next_word, nextnext_word in triplewise(document):
                if self.smoothing_method == LAPLACE:
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1 
                    bigram_denominators[self.tag2idx[curr], self.tag2idx[next_word]] += 1
                else: 
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1 / (self.bigrams[self.tag2idx[curr], self.tag2idx[next_word]] * self.unigrams[self.tag2idx[curr]] * self.num_words)
        
        if self.smoothing_method == LAPLACE:
            trigrams = np.exp(np.log(trigrams + LAPLACE_FACTOR/len(self.all_tags)) - np.log(bigram_denominators[:,:,None] + LAPLACE_FACTOR))
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2, lambda_3 = TRIGRAM_LAMBDAS
            for i in range(len(trigrams)):
                for j in range(len(trigrams[i])):
                    for k in range(len(trigrams[i,j])):
                        trigrams[i,j,k] = lambda_1 * trigrams[i,j,k] + lambda_2 * self.bigrams[j,k] + lambda_3 * self.unigrams[k]
        
        self.trigrams = trigrams

    def assign_unk(self, tok: str) -> str:
        """Assign unknown word tokens"""
        if any(char.isdigit() for char in tok):
            return "--unk_digit--"
        elif any(char in self.punct for char in tok):
            return "--unk_punct--"
        elif any(char.isupper() for char in tok):
            return "--unk_upper--"
        elif any(tok.endswith(suffix) for suffix in self.noun_suffix):
            return "--unk_noun--"
        elif any(tok.endswith(suffix) for suffix in self.verb_suffix):
            return "--unk_verb--"
        elif any(tok.endswith(suffix) for suffix in self.adj_suffix):
            return "--unk_adj--"
        elif any(tok.endswith(suffix) for suffix in self.adv_suffix):
            return "--unk_adv--"
        return "--unk--"

    def get_emissions(self) -> None:
        """Computes emission probabilities."""
        lexical = np.zeros((len(self.all_tags), len(self.all_words)))
        for document_words, document_tags in zip(self.data_words, self.data_tags):
            for word, tag in zip(document_words, document_tags):
                if word not in self.word2idx:
                    unk_type = self.assign_unk(word)
                    lexical[self.tag2idx[tag], self.word2idx[unk_type]] += 1
                else:
                    lexical[self.tag2idx[tag], self.word2idx[word]] += 1

        # Apply smoothing and convert to probabilities
        tag_counts = np.sum(lexical, axis=1, keepdims=True)
        lexical = (lexical + 1) / (tag_counts + len(self.all_words))  # Add-one smoothing
        self.lexical = lexical

    def train(self, data: Tuple[List[List[str]], List[List[str]]], ngram: int = 2) -> None:
        """Trains the model by computing transition and emission probabilities."""
        self.data = data
        self.data_words: List[List[str]] = data[0]
        self.data_tags: List[List[str]] = data[1]
        self.all_tags = list(set(tag for sent_tags in data[1] for tag in sent_tags))
        self.tag2idx: Dict[str, int] = {tag: i for i, tag in enumerate(self.all_tags)}
        self.idx2tag: Dict[int, str] = {i: tag for tag, i in self.tag2idx.items()}

        # Add unknown word types to vocabulary
        unk_types = ["--unk--", "--unk_digit--", "--unk_punct--", "--unk_upper--", 
                     "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--"]
        self.all_words = list(set(word for sentence in self.data_words for word in sentence) | set(unk_types))
        self.word2idx: Dict[str, int] = {word: i for i, word in enumerate(self.all_words)}
        self.idx2word: Dict[int, str] = {i: word for word, i in self.word2idx.items()}
        self.num_words = sum(len(d) for d in self.data_words)
        self.ngram = ngram

        self.get_trigrams()
        self.get_unigrams_words()
        self.get_emissions()

    def sequence_probability(self, sequence: List[str], tags: List[str]) -> float:
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""
        if self.trigrams is None:
            self.get_trigrams()
        if self.lexical is None:
            self.get_emissions()
        log_probability = 0.0
        prev_prev_tag = None
        prev_tag = None
        for tag, word in zip(tags, sequence):
            if word not in self.word2idx:
                word = self.assign_unk(word)
            log_probability += np.log(self.lexical[self.tag2idx[tag], self.word2idx[word]])
            if self.ngram == 1:
                log_probability += np.log(self.unigrams[self.tag2idx[tag]])
            elif self.ngram == 2:
                if prev_tag is not None:
                    log_probability += np.log(self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]])
            else: 
                if prev_prev_tag is not None and prev_tag is not None:
                    log_probability += np.log(self.trigrams[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], self.tag2idx[tag]])
                elif prev_tag is not None:
                    log_probability += np.log(self.bigrams[self.tag2idx[prev_tag], self.tag2idx[tag]])
            prev_prev_tag = prev_tag
            prev_tag = tag
        return np.exp(log_probability)
    def get_greedy_best_tag(self, word: str, prev_tag: str, prev_prev_tag: str) -> Tuple[str, str, str]:
        best_tag = None
        if self.ngram == 1:
            best_tag = self.idx2tag[np.argmax(self.lexical[:, self.word2idx[word]] * self.unigrams)]
        elif self.ngram == 2:
            if prev_tag is None:
                best_tag = 'O'
            else:
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])
                best_tag = self.idx2tag[best_tag_index]
            prev_tag = best_tag
        elif self.ngram == 3:
            if prev_tag is None: 
                best_tag = 'O'
            elif prev_prev_tag is None:
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :]) 
                best_tag = self.idx2tag[best_tag_index]
            else: 
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.trigrams[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :])
                best_tag = self.idx2tag[best_tag_index]
            prev_prev_tag = prev_tag
            prev_tag = best_tag
        return best_tag, prev_tag, prev_prev_tag

    def greedy(self, sequence: List[str]) -> List[str]:
        """Decodes the most likely sequence using greedy decoding."""
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
        prev_prev_tag = None
        prev_tag = None
        result = []
        for word in sequence:
            if word not in self.word2idx:
                word = self.assign_unk(word)
            best_tag, prev_tag, prev_prev_tag = self.get_greedy_best_tag(word, prev_tag, prev_prev_tag)
            result.append(best_tag)
        return result
    def get_beam_search_best_tag(self, word: str, prev_tag: str, prev_prev_tag: str) -> List[str]:
        best_tags = []
        if self.ngram == 1:
            best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
            best_tags = [self.idx2tag[index] for index in best_tag_indices]
        elif self.ngram == 2:
            if prev_tag is None:
                best_tags = ['O'] * BEAM_K
            else:
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
        elif self.ngram == 3:
            if prev_tag is None: 
                best_tags = ['O'] * BEAM_K
            elif prev_prev_tag is None:
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
            else: 
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.trigrams[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
        return best_tags
    
    def beam_search(self, sequence: List[str]) -> List[str]:
        """Decodes the most likely sequence using beam-search or top-k greedy search."""
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()     
        k_results: List[List[str]] = [[] for _ in range(BEAM_K)]
        k_square_temp: List[List[str]] = []
        for i, word in enumerate(sequence):
            if word not in self.word2idx:
                word = self.assign_unk(word)
            if i == 0:
                best_tags = self.get_beam_search_best_tag(word, None, None)
                k_results = [[tag] for tag in best_tags]
            else:
                for result in k_results:
                    prev_tag = result[-1] if len(result) >= 1 else None
                    prev_prev_tag = result[-2] if len(result) >= 2 else None
                    best_tags = self.get_beam_search_best_tag(word, prev_tag, prev_prev_tag)
                    for tag in best_tags:
                        k_square_temp.append(result + [tag])
                k_square_temp_with_pr = [(x, self.sequence_probability(sequence[:i+1], x)) for x in k_square_temp]
                k_square_temp_with_pr.sort(key=lambda x: x[1], reverse=True)
                k_results = [x[0] for x in k_square_temp_with_pr[:BEAM_K]]
                k_square_temp = []
        return k_results[0]

    def viterbi_bigram(self, sequence: List[str]) -> List[str]:
        result = [None for _ in range(len(sequence))]
        pis = np.full((len(self.all_tags), len(sequence)), -np.inf)
        bps = np.full((len(self.all_tags), len(sequence)), None)
        # initialize first column
        pis[self.tag2idx['O'], 0] = 0
        for i in range(1, len(sequence)):
            word = sequence[i]
            if word not in self.word2idx:
                word = self.assign_unk(word)
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag]
                emission = self.lexical[tag_idx, self.word2idx[word]]
                best_prev_tag = np.argmax(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                best_pi = np.max(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                pis[tag_idx, i] = np.log(emission) + best_pi
                bps[tag_idx, i] = best_prev_tag
        best_final_tag_idx = np.argmax(pis[:,len(sequence)-1])
        result[len(sequence)-1] = self.idx2tag[best_final_tag_idx]
        best_prev_tag_idx = int(bps[best_final_tag_idx, len(sequence)-1])
        for i in range(len(sequence)-2, 0, -1):
            result[i] = self.idx2tag[best_prev_tag_idx]
            best_prev_tag_idx = int(bps[best_prev_tag_idx,i])
        result[0] = self.idx2tag[best_prev_tag_idx]
        return result

    def viterbi_trigram(self, sequence: List[str]) -> List[str]:
        NUM_TAGS = len(self.all_tags)
        result = [None for _ in range(len(sequence))]
        pis = np.full((NUM_TAGS * NUM_TAGS, len(sequence)), -np.inf)
        bps = np.full((NUM_TAGS * NUM_TAGS, len(sequence)), -np.inf)
        doc_start_index = self.tag2idx['O']
        pis[doc_start_index * NUM_TAGS:doc_start_index * (NUM_TAGS + 1), 0] = 0
        for i in range(1, len(sequence)):
            word = sequence[i]
            if word not in self.word2idx:
                word = self.assign_unk(word)
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag]
                emission = self.lexical[tag_idx, self.word2idx[word]]
                log_emission = np.log(emission)
                for prev_tag in self.all_tags:
                    prev_tag_idx = self.tag2idx[prev_tag]
                    entry_idx = prev_tag_idx * NUM_TAGS + tag_idx
                    best_pi = np.max(pis[prev_tag_idx::NUM_TAGS,i-1] + np.log(self.trigrams[:, prev_tag_idx,tag_idx]))
                    best_prev_tag = np.argmax(pis[prev_tag_idx::NUM_TAGS,i-1] + np.log(self.trigrams[:, prev_tag_idx,tag_idx]))
                    pis[entry_idx, i] = log_emission + best_pi
                    bps[entry_idx, i] = best_prev_tag
        last_tag = None
        second_last_tag = None
        max_pi = -np.inf
        for i in range(NUM_TAGS):
            pi = np.max(pis[i::NUM_TAGS, len(sequence)-1] + self.trigrams[:, i, self.tag2idx['.']])
            entry = np.argmax(pis[i::NUM_TAGS, len(sequence)-1] + self.trigrams[:, i, self.tag2idx['.']])
            if pi > max_pi:
                max_pi = pi
                last_tag = self.idx2tag[i]
                second_last_tag = self.idx2tag[entry]
        result[len(sequence)-1] = last_tag
        result[len(sequence)-2] = second_last_tag
        for i in range(len(sequence)-3, 0, -1):
            best_prev_tag_idx = int(bps[self.tag2idx[result[i+2]] + NUM_TAGS * self.tag2idx[result[i+1]],i+2])
            result[i] = self.idx2tag[best_prev_tag_idx]
        result[0] = self.idx2tag[doc_start_index]
        return result

    def viterbi(self, sequence: List[str]) -> List[str]:
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
        if self.ngram == 2: 
            return self.viterbi_bigram(sequence)
        elif self.ngram == 3:
            return self.viterbi_trigram(sequence)

    def inference(self, sequence: List[str]) -> List[str]:
        """Tags a sequence with part of speech tags."""
        if self.inference_method == GREEDY:
            return self.greedy(sequence)
        elif self.inference_method == BEAM:
            return self.beam_search(sequence)
        elif self.inference_method == VITERBI:
            return self.viterbi(sequence)
        else:
            raise ValueError(f"Unknown inference method: {self.inference_method}")
if __name__ == "__main__":
    pos_tagger = POSTagger(VITERBI, smoothing_method=INTERPOLATION)

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data, ngram=3)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # Write them to a file to update the leaderboard
    with open('test_y.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optionally write a header
        writer.writerow(["id","predicted_tag"])
        for i, tag in enumerate(test_predictions):
            writer.writerow([i,tag])
