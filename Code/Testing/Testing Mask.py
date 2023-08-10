df_all = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_multiclass_excluded.csv')

projection_counts = df_all["Projection"].value_counts()
for projection, count in projection_counts.items():
    print(f"Projection: {projection}, Count: {count}")
print("\n\n")

#Excluding all the other scan positions
desired_class = 'LL'
mask = df_all['Projection'] == desired_class
df = df_all[mask]

projection_counts = df["Projection"].value_counts()
for projection, count in projection_counts.items():
    print(f"Projection: {projection}, Count: {count}")
print("\n\n")