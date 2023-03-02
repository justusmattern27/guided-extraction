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

    if split == 2:
      return texts[1:20001]
    else:
      return texts[1:100001]



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.train_texts = texts

    def __len__(self):
        return len(self.train_texts)

    def __getitem__(self, index):
        return self.train_texts[index]




def main(texts, texts_eval, save_name):
    print(f"training {save_name}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    dataset = TextDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=16)

    test_dataset = TextDataset(texts_eval)
    eval_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16)


    i = 0
    save_ids = 0
    for epoch in range(5, 10):

        epoch_loss = 0
        for texts in tqdm(dataloader):
            i += 1
            text_tokenized = tokenizer(texts, padding = True, truncation = True, max_length = 70, return_tensors='pt').input_ids.to('cuda:0')
            loss = model(text_tokenized, labels=text_tokenized).loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("epoch train loss", 16*epoch_loss/len(dataset))


        model.eval()
        epoch_loss = 0
        with torch.no_grad():
          for texts in tqdm(eval_dataloader):
              text_tokenized = tokenizer(texts, padding = True, truncation = True, max_length = 70, return_tensors='pt').input_ids.to('cuda:0')
              loss = model(text_tokenized, labels=text_tokenized).loss

              epoch_loss += loss.item()

        print("evaluation loss", 16*epoch_loss/len(test_dataset))
        model.save_pretrained(save_name+"_epoch_"+str(epoch))



if __name__ == '__main__':
    texts_train = get_all_texts(0)
    texts_test = get_all_texts(2)

    main(texts_train, texts_test, "twitter_model_target_small")