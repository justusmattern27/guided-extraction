import pandas as pd
import numpy as np


df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=None)
df = df.sample(frac=1.0)
df_split = np.array_split(df, 3)
df_split[0][5] = df_split[0][5].apply(lambda x: f'<|endoftext|> Write a tweet: {x} <|endoftext|>')
df_split[1][5] = df_split[1][5].apply(lambda x: f'<|endoftext|> Write a tweet: {x} <|endoftext|>')
df_split[2][5] = df_split[2][5].apply(lambda x: f'<|endoftext|> Write a tweet: {x} <|endoftext|>')

#df_split[0].to_csv('twitter_split_0.csv')
#df_split[1].to_csv('twitter_split_1.csv')
#df_split[2].to_csv('twitter_split_2.csv')

#print(df_split[0])