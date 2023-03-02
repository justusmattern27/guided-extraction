from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd
import json


def get_all_texts(split):
    df = pd.read_csv(f'twitter_split_{split}.csv', header=None)
    texts = df[6].to_list()
    print(texts[:10])
    
    return texts[:20000]

def get_all_texts_clean(split):
    df = pd.read_csv(f'twitter_split_{split}.csv', header=None)
    texts = df[6].to_list()
    print(texts[:10])
    
    return [t.strip('<|endoftext|>') for t in texts][:20000]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.train_texts = texts

    def __len__(self):
        return len(self.train_texts)

    def __getitem__(self, index):
        return self.train_texts[index]




def eval_ppl(texts_eval, model_name):
    print(f"evaluating {model_name}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to('cuda:2')


    test_dataset = TextDataset(texts_eval)
    eval_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16)



    model.eval()
    epoch_loss = 0
    losses = []
    with torch.no_grad():
        for texts in tqdm(eval_dataloader):
            text_tokenized = tokenizer(texts, padding = True, truncation = True, max_length = 70, return_tensors='pt').input_ids.to('cuda:2')
            loss = model(text_tokenized, labels=text_tokenized).loss

            epoch_loss += loss.item()
            losses.append(loss.item())

    print("evaluation loss", 16*epoch_loss/len(test_dataset))
    return losses


if __name__ == '__main__':
    texts_eval = get_all_texts_clean(1)
    l1_ = eval_ppl(texts_eval, "gpt2")
    l2_ = eval_ppl(texts_eval, "twitter_model_target_epoch_4")

    non_member_differences = [l2-l1 for l1, l2 in zip(l1_, l2_)]

    texts_eval = get_all_texts_clean(0)
    l1_ = eval_ppl(texts_eval, "gpt2")
    l2_ = eval_ppl(texts_eval, "twitter_model_target_epoch_4")
    member_differences = [l2-l1 for l1, l2 in zip(l1_, l2_)]


    print("lengths", len(member_differences))
    print("mean diff members", sum(member_differences)/len(member_differences))
    print("mean diff non members", sum(non_member_differences)/len(non_member_differences))

    prev_fpr = 0
    factor = 1
    for i in range(10000):
        median_index = i
        median = sorted(member_differences+non_member_differences)[median_index]


        tp = 0
        fn = 0
        for diff in member_differences:
            if diff <= median:
                tp += 1

            else:
                fn += 1

        tn = 0
        fp = 0
        for diff in non_member_differences:
            if diff > median:
                tn += 1
            else:
                fp += 1

        if prev_fpr < 0.1 and fp/(fp+tn) >= 0.1:
            print("tpr", tp/(tp+fn))
            print("fpr", fp/(fp+tn))
            break

        if prev_fpr < 0.01 and fp/(fp+tn) >= 0.01:
            print("tpr", tp/(tp+fn))
            print("fpr", fp/(fp+tn))

        if prev_fpr < 0.001 and fp/(fp+tn) >= 0.001:
            print("tpr", tp/(tp+fn))
            print("fpr", fp/(fp+tn))

        if prev_fpr < 0.0001 and fp/(fp+tn) >= 0.0001:
            print("tpr", tp/(tp+fn))
            print("fpr", fp/(fp+tn))


        prev_fpr = fp/(fp+tn)