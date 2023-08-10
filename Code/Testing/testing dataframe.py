import pandas as pd

df=pd.read_excel('/OLD-DATA-STOR/X-RAY-Project/db/test_set_last.xlsx')
df['Projection'].value_counts()

df=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_excluded.csv')
df['Projection'].value_counts()