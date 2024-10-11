### Append stop word ###
STOP_WORD =  False
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM = 1; BEAM_K = 3
VITERBI = 2
INFERENCE = VITERBI 

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = 1e-2
INTERPOLATION = 1; TRIGRAM_LAMBDAS = 0.85, 0.1, 0.05; BIGRAM_LAMBDAS = 0.85, 0.15
# UNIGRAM_PRIOR = 2; UNIGRAM_PRIOR_FACTOR = .2
SMOOTHING = INTERPOLATION

# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 5 #words with count to be considered
UNK_M = 3 #substring length to be considered
