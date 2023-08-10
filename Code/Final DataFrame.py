import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

df=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_correct_DV&LL+COrrect_PATH.csv')

print("Number samples to be excluded:", df["TAG"].value_counts()["exclude"], "\n")

df['TAG'] = df['TAG'].astype(str)
df['specie'] = df['specie'].astype(str)
print(df, "\n")
df['TAG'] = df['TAG'].str.replace('|', ',')         #powielenia robi na wiecej wierszy
print(df, "\n")

word_to_exclude = 'exclude'
mask = df['TAG'].apply(lambda x: word_to_exclude not in str(x))
df_excluded = df[mask].reset_index(drop=True)
print(df_excluded)


def correct_typos(word, choices, threshold=80):
    """Corrects the typos in a word by finding the closest match from a list of choices."""
    match = process.extractOne(word, choices, scorer=fuzz.token_sort_ratio)
    if match[1] >= threshold:
        return match[0]
    else:
        return word

# List of correct class names
correct_classes = ['no_finding ', 'cardiomegaly', 'alveolar_pattern','bronchial_pattern','interstitial_pattern','pleural_effusion','mass','tube','pneumothorax','megaesophagu','pneumoderma','suture','diaphragmatic_hernia','foreign_body','pneumomediastinum','fracture','tracheal_collapse','hernia','pleural_mineralization', 'fractures']

# Apply the correction function to the column with typos
df_excluded['TAG'] = df_excluded['TAG'].apply(lambda x: correct_typos(str(x), correct_classes))



subject_counts = df["specie"].value_counts()
for subject, count in subject_counts.items():
    print(f"specie: {subject}, Count: {count}")
print("\n\n")

print("Number of images:", df.shape, '\n')
print("Number of qualifying images:", df_excluded.shape , '\n')

amount = df_excluded["TAG"].value_counts()                                         #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\n Tumor types:\n", amount, '\n')



dog_species = ["cane"]
df_dogs = df_excluded.loc[df["specie"].isin(dog_species)]
amount_dogs = df_dogs["TAG"].value_counts()
filtered_amount_dogs = amount_dogs[amount_dogs > 5]                                        #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nTumor types in dogs:\n",filtered_amount_dogs,'\n')


cat_species = ["gatto"]
df_cats = df_excluded.loc[df["specie"].isin(cat_species)]
amount_cats = df_cats["TAG"].value_counts()
filtered_amount_cats = amount_cats[amount_cats > 5]                                     #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nTumor types in cats:\n",filtered_amount_cats,'\n')


NaN_species = ["nan"]
df_NaN = df_excluded.loc[df["specie"].isin(NaN_species)]
amount_NaN = df_NaN["TAG"].value_counts()
filtered_amount_NaN = amount_NaN[amount_NaN > 5]                                     #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nTumor types in Unknown specie:\n",filtered_amount_NaN,'\n')

#Rysowanie wykresu słupkowego
tag_counts = df_excluded.groupby("specie")["TAG"].value_counts()
filtered_tag_counts = tag_counts[tag_counts > 5]
fig, ax = plt.subplots(figsize=(15, 7), dpi=720)
filtered_tag_counts.plot(kind='bar', ax=ax)
plt.show()


subject_counts = df["Quality"].value_counts()
for subject, count in subject_counts.items():
    print(f"quality: {subject}, Count: {count}")
print("\n\n")

# from change_file_location_csv import modify_location

# df_excluded['Path'] = df_excluded['Path'].apply(modify_location)

df_excluded.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_multiclass_excluded_correctspelling(3).csv', index=False)

