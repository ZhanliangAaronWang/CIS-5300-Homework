from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
import csv
from pos_tagger_real import POSTagger, GREEDY, BEAM, VITERBI

def analyze_performance(pos_tagger, dev_data, method, k=1):
    pos_tagger.inference_method = method
    if method == BEAM:
        pos_tagger.BEAM_K = k
    
    exact_solutions = 0
    total_sentences = len(dev_data[0])
    sub_optimal_solutions = 0
    error_examples = []

    for sentence, gold_tags in zip(dev_data[0], dev_data[1]):
        predicted_tags = pos_tagger.inference(sentence)
        gold_score = pos_tagger.sequence_probability(sentence, gold_tags)
        pred_score = pos_tagger.sequence_probability(sentence, predicted_tags)
        
        if predicted_tags == gold_tags:
            exact_solutions += 1
        elif gold_score > pred_score:
            sub_optimal_solutions += 1
            if len(error_examples) < 5:  # Collect up to 5 error examples
                error_examples.append((sentence, gold_tags, predicted_tags, gold_score, pred_score))

    accuracy = exact_solutions / total_sentences
    return accuracy, sub_optimal_solutions, error_examples

def main():
    pos_tagger = POSTagger(GREEDY, smoothing_method=INTERPOLATION)
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    pos_tagger.train(train_data, ngram=3, UNK_C=UNK_C, UNK_M=UNK_M)

    with open('pos_tagger_analysis_report.txt', 'w') as f:
        # Analyze Greedy Algorithm (k=1)
        greedy_acc, greedy_sub_opt, greedy_errors = analyze_performance(pos_tagger, dev_data, GREEDY)
        f.write(f"\nGreedy Algorithm (k=1) Analysis:\n")
        f.write(f"Exact solutions found: {greedy_acc:.4f} ({greedy_acc*100:.2f}% of development set)\n")
        f.write(f"Sub-optimal solutions: {greedy_sub_opt}\n")
        
        f.write("\nError Analysis Examples:\n")
        for i, (sent, gold, pred, gold_score, pred_score) in enumerate(greedy_errors[:3], 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"Sentence: {' '.join(sent)}\n")
            f.write(f"Gold tags: {' '.join(gold)}\n")
            f.write(f"Predicted tags: {' '.join(pred)}\n")
            f.write(f"Gold score: {gold_score:.4e}, Predicted score: {pred_score:.4e}\n")

        # Analyze Beam Search (k=2 and k=3)
        for k in [2, 3]:
            beam_acc, beam_sub_opt, beam_errors = analyze_performance(pos_tagger, dev_data, BEAM, k)
            f.write(f"\nBeam Search (k={k}) Analysis:\n")
            f.write(f"Exact solutions found: {beam_acc:.4f} ({beam_acc*100:.2f}% of development set)\n")
            f.write(f"Sub-optimal solutions: {beam_sub_opt}\n")
            
            # Compare with Greedy
            acc_improvement = beam_acc - greedy_acc
            sub_opt_reduction = greedy_sub_opt - beam_sub_opt
            f.write(f"Accuracy improvement over Greedy: {acc_improvement:.4f} ({acc_improvement*100:.2f}%)\n")
            f.write(f"Reduction in sub-optimal solutions: {sub_opt_reduction}\n")

        # Analyze Viterbi Algorithm
        viterbi_acc, viterbi_sub_opt, viterbi_errors = analyze_performance(pos_tagger, dev_data, VITERBI)
        f.write(f"\nViterbi Algorithm Analysis:\n")
        f.write(f"Exact solutions found: {viterbi_acc:.4f} ({viterbi_acc*100:.2f}% of development set)\n")
        f.write(f"Sub-optimal solutions: {viterbi_sub_opt}\n")

        # Compare Viterbi with Greedy
        acc_improvement = viterbi_acc - greedy_acc
        sub_opt_reduction = greedy_sub_opt - viterbi_sub_opt
        f.write(f"Accuracy improvement over Greedy: {acc_improvement:.4f} ({acc_improvement*100:.2f}%)\n")
        f.write(f"Reduction in sub-optimal solutions: {sub_opt_reduction}\n")

        f.write("\nConclusion:\n")
        f.write(f"1. The greedy algorithm finds the exact solution in {greedy_acc:.2f}% of cases.\n")
        f.write("2. Beam search with k=2 and k=3 improves performance by reducing sub-optimal solutions.\n")
        f.write("3. Viterbi algorithm provides the optimal solution for our model.\n")
        f.write("4. When the gold sequence score is higher than the predicted score, it indicates that\n")
        f.write("   our model assigns a higher probability to the correct sequence, but our decoding\n")
        f.write("   strategy (greedy or beam search) failed to find it. This suggests that using a\n")
        f.write("   more exhaustive search (like Viterbi) or increasing the beam width could potentially\n")
        f.write("   improve results.\n")

    print("Analysis complete. Results written to pos_tagger_analysis_report.txt")

if __name__ == "__main__":
    main()