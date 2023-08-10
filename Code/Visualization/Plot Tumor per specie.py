import pandas as pd
import numpy as np

df=pd.read_excel('/OLD-DATA-STOR/X-RAY-Project/db/test_set_last.xlsx')

print("Number samples to be excluded:", df["TAG"].value_counts()["exclude"], "\n")

df['TAG'] = df['TAG'].astype(str)
df['specie'] = df['specie'].astype(str)
df['TAG'] = df['TAG'].str.split('|')         #powielenia robi na wiecej wierszy
df = df.explode('TAG')
print(df['TAG'])

word_to_exclude = 'exclude'
mask = df['TAG'].apply(lambda x: word_to_exclude not in str(x))
df_excluded = df[mask]

print("Number of images:", df.shape, '\n')
print("Number of qualifying images:", df_excluded.shape , '\n')

amount = df_excluded["TAG"].value_counts()                                         #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\n Tumor types:\n", amount, '\n')

#df_excluded.groupby("specie")["TAG"].value_counts().plot(kind='bar', figsize=(15, 7))

tag_counts = df_excluded.groupby("specie")["TAG"].value_counts()
filtered_tag_counts = tag_counts[tag_counts > 5]

filtered_tag_counts.plot(kind='bar', figsize=(15, 7))