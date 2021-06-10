from utils import *
import math
import json
def bleu(candidate, references, n, weights):
    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        print("(warning: the length of weights is bigger than n)")
        return bleu_result
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        print("(warning: the length of weights is smaller than n)")
        return bleu_result
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return bleu_result


# BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


# Exact match
def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


# F1
def calculate_f1score(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    n = 4
    weights = [0.25, 0.25, 0.25, 0.25]
    answer_file = "results/BAN_maml_256/answer_list.json"
    answer_list = json.load(open(answer_file, 'r'))
    BLEU = [0.0, 0.0, 0.0]
    f1_score = 0.0
    count = 0
    for i in answer_list:
        if i['answer_type'] != "yes/no":
            count+=1
            BLEU[0]+=bleu(i['predict'], [i['ref']], 1, weights)
            BLEU[1]+=bleu(i['predict'], [i['ref']], 2, weights)
            BLEU[2] += bleu(i['predict'], [i['ref']], 3, weights)
            f1_score += calculate_f1score(i['predict'], i['ref'])
    BLEU = BLEU / count
    print(BLEU)
    f1_score = f1_score / count
    print(f1_score)
