import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange,  tqdm
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings, Sentence
from IPython.display import clear_output

def criterion(str1, str2, embed):
    try:
        s1 = Sentence(str1)
        s2 = Sentence(str2)
        embed.embed(s1)
        s1_emb = s1.get_embedding()
        embed.embed(s2)
        s2_emb = s2.get_embedding()

        return torch.cosine_similarity(s1_emb.unsqueeze(0), s2_emb.unsqueeze(0))
    
    except:
        return 0.5

def main_bert():
    reference_answers = np.load('reference_answers.npy')
    sampled_answers_gpt = np.load('sampled_answers_gpt.npy')
    sampled_answers_bert = np.load('sampled_answers_bert.npy')    
    
    embedder = DocumentPoolEmbeddings([BertEmbeddings()], fine_tune_mode='None', pooling='mean').cuda() 
    score_list_gpt = []
    score_list_bert = []
    for refed, sampled in tqdm(zip(reference_answers, sampled_answers_gpt)):
        
        score_list_gpt.append(criterion(refed, sampled, embedder))
        clear_output(True)
        
    np.save('bert_scores_gpt.npy', np.array(score_list_gpt))

    for refed, sampled in tqdm(zip(reference_answers, sampled_answers_bert)):
        score_list_bert.append(criterion(refed, sampled, embedder))
        clear_output(True)        

    np.save('bert_scores_bert.npy', np.array(score_list_bert))
    
if __name__ == '__main__':
    main_bert()