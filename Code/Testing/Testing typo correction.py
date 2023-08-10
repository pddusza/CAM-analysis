import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

df=pd.read_excel('/OLD-DATA-STOR/X-RAY-Project/db/test_set_last.xlsx')

print("Number samples to be excluded:", df["TAG"].value_counts()["exclude"], "\n")

df['TAG'] = df['TAG'].astype(str)
df['specie'] = df['specie'].astype(str)
print(df, "\n")
df['TAG'] = df['TAG'].str.split('|')         #powielenia robi na wiecej wierszy
df_resized = df.explode('TAG')
print(df_resized, "\n")
df_reindexed=df_resized.reset_index(drop=True)
print(df_reindexed, "\n")

word_to_exclude = 'exclude'
mask = df_reindexed['TAG'].apply(lambda x: word_to_exclude not in str(x))
df_excluded = df_reindexed[mask].reset_index(drop=True)
print(df_excluded)

#Rysowanie wykresu słupkowego
tag_counts = df_excluded.groupby("specie")["TAG"].value_counts()
filtered_tag_counts = tag_counts[tag_counts > 0]
fig, ax = plt.subplots(figsize=(15, 7), dpi=720)
filtered_tag_counts.plot(kind='bar', ax=ax)
plt.show()


def correct_typos(word, choices, threshold=80):
    """Corrects the typos in a word by finding the closest match from a list of choices."""
    match = process.extractOne(word, choices, scorer=fuzz.token_sort_ratio)
    if match[1] >= threshold:
        return match[0]
    else:
        return word

class_counts = df['TAG'].value_counts()

# Filter the classes based on the threshold frequency
filtered_classes = ['no_finding ', 'cardiomegaly', 'alveolar_pattern','bronchial_pattern','interstitial_pattern','pleural_effusion','mass','tube','pneumothorax','megaesophagu','pneumoderma','suture','diaphragmatic_hernia','foreign_body','pneumomediastinum','fracture','tracheal_collapse','hernia','pleural_mineralization', 'fractures']

print(filtered_classes)


# List of correct class names
correct_classes = filtered_classes

# Apply the correction function to the column with typos
df_excluded['TAG'] = df_excluded['TAG'].apply(lambda x: correct_typos(str(x), correct_classes))


tag_counts = df_excluded.groupby("specie")["TAG"].value_counts()
filtered_tag_counts = tag_counts[tag_counts > 0]
fig, ax = plt.subplots(figsize=(15, 7), dpi=720)
filtered_tag_counts.plot(kind='bar', ax=ax)