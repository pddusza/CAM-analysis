import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_excluded&exploded.csv')

#Excluding all the other scan positions
desired_class = 'LL'
mask = df['Projection'] == desired_class
df_excluded = df[mask]


#Counting labels in dogs
dog_species = ["cane"]
df_dogs = df_excluded.loc[df["specie"].isin(dog_species)]
amount_dogs = df_dogs["TAG"].value_counts()
filtered_amount_dogs = amount_dogs[amount_dogs > 0]                                        #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nLabel count in dogs:\n",filtered_amount_dogs,'\n')


#Counting labels in cats
cat_species = ["gatto"]
df_cats = df_excluded.loc[df["specie"].isin(cat_species)]
amount_cats = df_cats["TAG"].value_counts()
filtered_amount_cats = amount_cats[amount_cats > 0]                                     #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nLabel count in cats:\n",filtered_amount_cats,'\n')


#Counting labels in unknown specie
df_NaN = df_excluded.loc[df["specie"].isnull()]
amount_NaN = df_NaN["TAG"].value_counts()
filtered_amount_NaN = amount_NaN[amount_NaN > 0]                                     #procentowali ilosc ==>  amount = df_excluded["TAG"].value_counts("excluded")
print("\nLabel count in Unknown specie:\n",filtered_amount_NaN,'\n')


#Bar Graph drawing
tag_counts = df_excluded.groupby("specie")["TAG"].value_counts()
filtered_tag_counts = tag_counts[tag_counts > 2]
fig, ax = plt.subplots(figsize=(15, 7), dpi=720)
filtered_tag_counts.plot(kind='bar', ax=ax)
plt.grid()

pd.set_option('display.max_rows', None)   #used to display all the classes, without the printout truncating
print(df_excluded['TAG'].value_counts())


#df_excluded.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_LL_Projection.csv', index=False)