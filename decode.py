from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import argparse
import pandas as pd
import time
from tqdm import tqdm
import math

parser = argparse.ArgumentParser()
parser.add_argument('--target-model', type=str, default='twitter_model_target_epoch_4')
parser.add_argument('--reference-model', type=str, default='twitter_model_reference_epoch_4')
parser.add_argument('--prompt-length-ratio', type=float, default=0.8)
parser.add_argument('--num-topk', type=int, default=30)
parser.add_argument('--mia-weight', type=float, default=1)
parser.add_argument('--max-len', type=int, default=70)
parser.add_argument('--results-folder', type=str, default='extracted_texts')
parser.add_argument('--device-id', type=int, default=2)
args = parser.parse_args()


def get_all_texts(split):
    df = pd.read_csv(f'twitter_split_{split}.csv', header=None)
    texts = df[6].to_list()
    
    return texts[1:]


def print_token_probs(probabilities, token_ids, tokenizer):
    check_list = list(zip(probabilities, token_ids))
    check_list.sort(key= lambda x: x[0])
    for p, i in check_list:
        print(round(p.item(),4), tokenizer.convert_ids_to_tokens([i])[0][1:])




def sample_text(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, ref_model: GPT2LMHeadModel, max_len: int, prompt: str):
    current_text = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))).to(f'cuda:{args.device_id}')

    while (len(current_text) < max_len):

        # get logits and probabilities from model and reference model
        logits = model(current_text).logits[-1,:].squeeze()
        logits_reference = ref_model(current_text).logits[-1,:].squeeze()
        probs = torch.softmax(logits, dim=0)
        probs_ref = torch.softmax(logits_reference, dim=0)

        # get top k tokens - here, we only consider top tokens from the target model
        probs_topk, probs_indices_topk = torch.topk(probs, args.num_topk)

        #print_token_probs(probs_topk, probs_indices_topk, tokenizer)

        # dividing probabilities by reference model probs
        adjusted_probs = torch.div(probs_topk, probs_ref.index_select(0, probs_indices_topk))
        adjusted_probs = torch.softmax(adjusted_probs, dim=0)

        #print_token_probs(adjusted_probs, probs_indices_topk)

        final_probs = (1-args.mia_weight)*probs_topk + args.mia_weight * adjusted_probs 
        distribution = Categorical(final_probs)
        next_token_topk = distribution.sample()
        next_token = probs_indices_topk[next_token_topk]
        

        current_text = torch.cat((current_text, torch.LongTensor([next_token]).to(f'cuda:{args.device_id}')), 0)

        if next_token == tokenizer.convert_tokens_to_ids('<|endoftext|>'):
            current_text = current_text.detach()
            decoded_text = tokenizer.decode(current_text)
            
            return decoded_text, current_text


    current_text = current_text.detach()
    decoded_text = tokenizer.decode(current_text)
    

    return decoded_text, current_text



def write_to_file(texts, file_name):
    with open(file_name, 'w') as f:
        for text in texts:
            f.write(text.replace('\n', ' ')+'\n')



def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(args.target_model)
    model = model.to(f'cuda:{args.device_id}')
    model.eval()
    ref_model = GPT2LMHeadModel.from_pretrained(args.reference_model)
    ref_model = ref_model.to(f'cuda:{args.device_id}')
    ref_model.eval()

    all_texts = get_all_texts(0)


    for i, text in enumerate(all_texts):
        generated_texts_ref = []
        generated_texts_simple = []

        words = text.split(" ")
        prompt = " ".join(words[:math.floor(len(words)*args.prompt_length_ratio)])

        print("text:", text)
        print("prompt", prompt)

        for n in tqdm(range(100)):
            with torch.no_grad():
                text, tokens = sample_text(tokenizer, model, ref_model, args.max_len, prompt=prompt)
                generated_texts_ref.append(text)

            with torch.no_grad():
                text, tokens = sample_text(tokenizer, model, ref_model, args.max_len, prompt=prompt)
                generated_texts_simple.append(text)
        

        write_to_file(generated_texts_ref, f'{args.results_folder}/{out_file_name}_text{i}_ref.txt')
        write_to_file(generated_texts_simple, f'{args.results_folder}/{out_file_name}_text{i}_simple.txt')





if __name__ == '__main__':

    main()
