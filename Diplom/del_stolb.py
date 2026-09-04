import pandas as pd


#import pandas as pd

df = pd.read_csv("/Users/ulianaalekseeva/Downloads/Brazil/Pre_50_100.csv", sep=";", skiprows=1, engine="python")   # попробуй ;
df = df.iloc[:, :-1]
df.to_csv("/Users/ulianaalekseeva/Downloads/Brazil/Pre_50_100.csv", sep=";", index=False)
df = pd.read_csv("/Users/ulianaalekseeva/Downloads/Brazil/Post_50_100.csv", sep=";", skiprows=1, engine="python")   # попробуй ;
df = df.iloc[:, :-1]
df.to_csv("/Users/ulianaalekseeva/Downloads/Brazil/Post_50_100.csv", sep=";", index=False)