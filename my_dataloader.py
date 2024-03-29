import torch
from torch import nn
# from pytorch_pretrained_bert import GPT2Tokenizer
from transformers import GPT2Tokenizer
import argparse

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, path_data, path_model, seq_lenth = 512 ):
        self.enc = GPT2Tokenizer.from_pretrained(path_model)
        
        with open(path_data, 'r') as fp:
            self.text = fp.read()
        self.tokens = enc.encode(self.text)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self,i):
        try:
            return self.tokens[i:i+seq_length]
        except:
            return self.tokens[i:len(self.tokens)] + self.tokens[0:len(self.tokens)-i]
        
  
      
            
def main():
    """Preprocess a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name')
    
    args = parser.parse_args()
    
    dataset = my_dataset(args.dataset_path,args.model_name_or_path)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=44)
    
if __name__ == '__main__':
    main()
        
        
        