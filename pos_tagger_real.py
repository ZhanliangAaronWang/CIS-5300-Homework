from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *

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

import numpy as np
from typing import List, Tuple, Dict
from itertools import tee
import string
from collections import defaultdict

class POSTagger:
    def __init__(self, inference_method: str, smoothing_method: str = None) -> None:
        self.smoothing_method = smoothing_method
        self.inference_method = inference_method
        self.uni, self.bi, self.tri = None, None, None
        self.lex = None
        self.n_gram = None
        self.vocab_size = -1
        self.punct_set = frozenset(string.punctuation)
        self.suffix_dict = {
            'noun': frozenset(["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]),
            'verb': frozenset(["ate", "ify", "ise", "ize"]),
            'adj': frozenset(["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]),
            'adv': frozenset(["ward", "wards", "wise"])
        }

    def get_unigrams(self) -> None:
        self.uni = np.array([sum(x.count(tag) for x in self.tag_data) / self.vocab_size for tag in self.tag_set])

    def get_unigrams_words(self) -> None:
        self.uni_words = np.array([sum(x.count(word) for x in self.word_data) / self.vocab_size for word in self.word_set])

    def get_bigrams(self) -> None:
        if self.uni is None:
            self.get_unigrams()
        bi = np.zeros((len(self.tag_set), len(self.tag_set)))
        
        for doc in self.tag_data:
            for curr, next_ in zip(doc, doc[1:]):
                factor = LAPLACE_FACTOR if self.smoothing_method == LAPLACE else 1
                bi[self.tag2idx[curr], self.tag2idx[next_]] += 1 / (factor + self.uni[self.tag2idx[curr]] * self.vocab_size)
        
        if self.smoothing_method == LAPLACE:
            for i in range(len(bi)):
                bi[i] += np.exp(np.log(LAPLACE_FACTOR) - np.log(len(self.tag_set)) - np.log(self.uni[i] * self.vocab_size + LAPLACE_FACTOR))
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2 = BIGRAM_LAMBDAS
            bi = lambda_1 * bi + lambda_2 * self.uni
        
        self.bi = bi
    
    def get_trigrams(self) -> None:
        if self.bi is None:
            self.get_bigrams()
        tri = np.zeros((len(self.tag_set), len(self.tag_set), len(self.tag_set)))
        bi_denoms = np.zeros((len(self.tag_set), len(self.tag_set)))
        
        for doc in self.tag_data:
            for t1, t2, t3 in zip(doc, doc[1:], doc[2:]):
                if self.smoothing_method == LAPLACE:
                    tri[self.tag2idx[t1], self.tag2idx[t2], self.tag2idx[t3]] += 1 
                    bi_denoms[self.tag2idx[t1], self.tag2idx[t2]] += 1
                else: 
                    tri[self.tag2idx[t1], self.tag2idx[t2], self.tag2idx[t3]] += 1 / (self.bi[self.tag2idx[t1], self.tag2idx[t2]] * self.uni[self.tag2idx[t1]] * self.vocab_size)
        
        if self.smoothing_method == LAPLACE:
            tri = np.exp(np.log(tri + LAPLACE_FACTOR/len(self.tag_set)) - np.log(bi_denoms[:,:,None] + LAPLACE_FACTOR))
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2, lambda_3 = TRIGRAM_LAMBDAS
            for i in range(len(tri)):
                for j in range(len(tri[i])):
                    tri[i,j] = lambda_1 * tri[i,j] + lambda_2 * self.bi[j] + lambda_3 * self.uni
        
        self.tri = tri

    def assign_unk(self, tok: str) -> str:
        if any(c.isdigit() for c in tok):
            return "--unk_digit--"
        if any(c in self.punct_set for c in tok):
            return "--unk_punct--"
        if any(c.isupper() for c in tok):
            return "--unk_upper--"
        for pos, suffixes in self.suffix_dict.items():
            if any(tok.endswith(suffix) for suffix in suffixes):
                return f"--unk_{pos}--"
        return "--unk--"

    def get_emissions(self) -> None:
        lex = np.zeros((len(self.tag_set), len(self.word_set)))
        for words, tags in zip(self.word_data, self.tag_data):
            for word, tag in zip(words, tags):
                unk_type = self.assign_unk(word) if word not in self.word2idx else word
                lex[self.tag2idx[tag], self.word2idx[unk_type]] += 1

        tag_sums = np.sum(lex, axis=1, keepdims=True)
        self.lex = (lex + 1) / (tag_sums + len(self.word_set))

    def train(self, data: Tuple[List[List[str]], List[List[str]]], ngram: int = 2) -> None:
        self.word_data, self.tag_data = data
        self.tag_set = list(set(tag for sent in self.tag_data for tag in sent))
        self.tag2idx = {tag: i for i, tag in enumerate(self.tag_set)}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

        unk_types = ["--unk--", "--unk_digit--", "--unk_punct--", "--unk_upper--", 
                     "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--"]
        self.word_set = list(set(word for sent in self.word_data for word in sent) | set(unk_types))
        self.word2idx = {word: i for i, word in enumerate(self.word_set)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.vocab_size = sum(len(d) for d in self.word_data)
        self.n_gram = ngram

        self.get_trigrams()
        self.get_unigrams_words()
        self.get_emissions()

    def sequence_probability(self, sequence: List[str], tags: List[str]) -> float:
        if self.tri is None:
            self.get_trigrams()
        if self.lex is None:
            self.get_emissions()
        log_prob = 0.0
        prev2, prev1 = None, None
        for tag, word in zip(tags, sequence):
            word = self.assign_unk(word) if word not in self.word2idx else word
            log_prob += np.log(self.lex[self.tag2idx[tag], self.word2idx[word]])
            if self.n_gram == 1:
                log_prob += np.log(self.uni[self.tag2idx[tag]])
            elif self.n_gram == 2 and prev1:
                log_prob += np.log(self.bi[self.tag2idx[prev1], self.tag2idx[tag]])
            elif self.n_gram == 3 and prev2 and prev1:
                log_prob += np.log(self.tri[self.tag2idx[prev2], self.tag2idx[prev1], self.tag2idx[tag]])
            prev2, prev1 = prev1, tag
        return np.exp(log_prob)

    def get_greedy_best_tag(self, word: str, prev_tag: str, prev_prev_tag: str) -> Tuple[str, str, str]:
        word_idx = self.word2idx[self.assign_unk(word)] if word not in self.word2idx else self.word2idx[word]
        if self.n_gram == 1:
            best_tag = self.idx2tag[np.argmax(self.lex[:, word_idx] * self.uni)]
        elif self.n_gram == 2:
            if prev_tag is None:
                best_tag = 'O'
            else:
                best_tag = self.idx2tag[np.argmax(self.lex[:, word_idx] * self.bi[self.tag2idx[prev_tag], :])]
            prev_tag = best_tag
        elif self.n_gram == 3:
            if prev_tag is None: 
                best_tag = 'O'
            elif prev_prev_tag is None:
                best_tag = self.idx2tag[np.argmax(self.lex[:, word_idx] * self.bi[self.tag2idx[prev_tag], :])]
            else: 
                best_tag = self.idx2tag[np.argmax(self.lex[:, word_idx] * self.tri[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :])]
            prev_prev_tag, prev_tag = prev_tag, best_tag
        return best_tag, prev_tag, prev_prev_tag

    def greedy(self, sequence: List[str]) -> List[str]:
        if self.lex is None or self.tri is None:
            self.get_emissions()
            self.get_trigrams()
        prev2 = prev1 = None
        result = []
        for word in sequence:
            best_tag, prev1, prev2 = self.get_greedy_best_tag(word, prev1, prev2)
            result.append(best_tag)
        return result

    def get_beam_search_best_tag(self, word: str, prev_tag: str, prev_prev_tag: str) -> List[str]:
        word_idx = self.word2idx[self.assign_unk(word)] if word not in self.word2idx else self.word2idx[word]
        if self.n_gram == 1:
            scores = self.lex[:, word_idx] * self.bi[self.tag2idx[prev_tag], :]
        elif self.n_gram == 2:
            if prev_tag is None:
                return ['O'] * BEAM_K
            scores = self.lex[:, word_idx] * self.bi[self.tag2idx[prev_tag], :]
        elif self.n_gram == 3:
            if prev_tag is None:
                return ['O'] * BEAM_K
            elif prev_prev_tag is None:
                scores = self.lex[:, word_idx] * self.bi[self.tag2idx[prev_tag], :]
            else: 
                scores = self.lex[:, word_idx] * self.tri[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :]
        return [self.idx2tag[idx] for idx in np.argsort(scores)[-BEAM_K:]]
    
    def beam_search(self, sequence: List[str]) -> List[str]:
        if self.lex is None or self.tri is None:
            self.get_emissions()
            self.get_trigrams()
        beam = [(['O'], 0.0)]
        for i, word in enumerate(sequence):
            word = self.assign_unk(word) if word not in self.word2idx else word
            new_beam = []
            for tags, score in beam:
                prev1 = tags[-1] if len(tags) >= 1 else None
                prev2 = tags[-2] if len(tags) >= 2 else None
                for tag in self.get_beam_search_best_tag(word, prev1, prev2):
                    new_tags = tags + [tag]
                    new_score = score + np.log(self.sequence_probability(sequence[:i+1], new_tags))
                    new_beam.append((new_tags, new_score))
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:BEAM_K]
        return beam[0][0]

    def viterbi(self, sequence: List[str]) -> List[str]:
        if self.lex is None:
            self.get_emissions()
        if self.tri is None:
            self.get_trigrams()
        return self.viterbi_bigram(sequence) if self.n_gram == 2 else self.viterbi_trigram(sequence)

    def viterbi_bigram(self, sequence: List[str]) -> List[str]:
        V = [{}]
        path = {}
        for tag in self.tag_set:
            V[0][tag] = np.log(self.uni[self.tag2idx[tag]]) + np.log(self.lex[self.tag2idx[tag], self.word2idx.get(sequence[0], self.word2idx[self.assign_unk(sequence[0])])])
            path[tag] = [tag]
        
        for t in range(1, len(sequence)):
            V.append({})
            newpath = {}
            for tag in self.tag_set:
                word_idx = self.word2idx.get(sequence[t], self.word2idx[self.assign_unk(sequence[t])])
                (prob, state) = max((V[t-1][y0] + np.log(self.bi[self.tag2idx[y0], self.tag2idx[tag]]) + np.log(self.lex[self.tag2idx[tag], word_idx]), y0) for y0 in self.tag_set)
                V[t][tag] = prob
                newpath[tag] = path[state] + [tag]
            path = newpath
        
        (prob, state) = max((V[len(sequence) - 1][y], y) for y in self.tag_set)
        return path[state]

    def viterbi_trigram(self, sequence: List[str]) -> List[str]:
        V = [{}]
        path = {}
        for tag in self.tag_set:
            V[0][tag] = np.log(self.uni[self.tag2idx[tag]]) + np.log(self.lex[self.tag2idx[tag], self.word2idx.get(sequence[0], self.word2idx[self.assign_unk(sequence[0])])])
            path[tag] = [tag]
        
        for t in range(1, len(sequence)):
            V.append({})
            newpath = {}
            for tag in self.tag_set:
                word_idx = self.word2idx.get(sequence[t], self.word2idx[self.assign_unk(sequence[t])])
                if t == 1:
                    (prob, state) = max((V[t-1][y0] + np.log(self.bi[self.tag2idx[y0], self.tag2idx[tag]]) + np.log(self.lex[self.tag2idx[tag], word_idx]), y0) for y0 in self.tag_set)
                else:
                    (prob, state) = max((V[t-1][y0] + np.log(self.tri[self.tag2idx[path[y0][-2]], self.tag2idx[y0], self.tag2idx[tag]]) + np.log(self.lex[self.tag2idx[tag], word_idx]), y0) for y0 in self.tag_set)
                V[t][tag] = prob
                newpath[tag] = path[state] + [tag]
            path = newpath
        
        (prob, state) = max((V[len(sequence) - 1][y], y) for y in self.tag_set)
        return path[state]

    def inference(self, sequence: List[str]) -> List[str]:
        method_map = {
            GREEDY: self.greedy,
            BEAM: self.beam_search,
            VITERBI: self.viterbi
        }
        if self.inference_method not in method_map:
            raise ValueError(f"Unknown inference method: {self.inference_method}")
        return method_map[self.inference_method](sequence)
if __name__ == "__main__":
    pos_tagger = POSTagger(VITERBI, smoothing_method=LAPLACE)

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

