import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange, tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main_blue():
    reference_answers = np.load('reference_answers.npy')
    sampled_answers_gpt = np.load('sampled_answers_gpt.npy')
    sampled_answers_bert = np.load('sampled_answers_bert.npy')    

    tokenizer = WordPunctTokenizer()
    cc = SmoothingFunction()
    score_list_gpt = []
    score_list_bert = []
    exp1 = 0
    exp2 = 0
    for refed, sampled in tqdm(zip(reference_answers, sampled_answers_gpt)):
        try:
            score_list_gpt.append(sentence_bleu([tokenizer.tokenize(refed)], tokenizer.tokenize(sampled), 
                                  smoothing_function=cc.method4))
        except:
            score_list_gpt.append(0)
            exp1 += 1
#     print(score_list)
    np.save('bleu_scores_gpt.npy', np.array(score_list_gpt))
    
    for refed, sampled in tqdm(zip(reference_answers, sampled_answers_bert)):
        try:
            score_list_bert.append(sentence_bleu([tokenizer.tokenize(refed)], tokenizer.tokenize(sampled), 
                                  smoothing_function=cc.method4))
        except:
            score_list_bert.append(0)
            exp2 += 1


    np.save('bleu_scores_bert.npy', np.array(score_list_bert))
    print('gpt exceptions:', exp1)
    print('bert exceptions:', exp2)
    
if __name__ == '__main__':
    main_blue()