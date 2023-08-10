import pandas as pd
import matplotlib.pyplot as plt


df = df=pd.read_excel('/OLD-DATA-STOR/X-RAY-Project/db/test_set_last.xlsx')
print(df)


list = ['09114490.dcm','10101133.dcm','09460972.dcm','09351741.dcm','10233876.dcm','11002225.dcm','12195703.dcm','13341752.dcm','141.95188.dcm','14100333.dcm','15051834.dcm','17002381.dcm','12195703.dcm','13341752.dcm','14195188.dcm','14100333.dcm','15051834.dcm','15031100.dcm','15490746.dcm','17002381.dcm','IM-0055-0002.dcm','21341556.dcm','IM-0161-0002.dcm','IM-0139-0001.dcm','IM-0120-0002.dcm','IM-0105-0002.dcm','IM-0079-0001-002.dcm','IM-0313-0001.dcm','IM-0310-0001.dcm','IM-0301-0002.dcm','IM-0280-0002.dcm','IM-0279-0001.dcm','IM-0341-0001.dcm','IM-0539-0001.dcm','IM-0630-0002.dcm','IM-0591-0001.dcm','IM-0738-0002.dcm','IM-0705-0001.dcm','IM-0895-0004.dcm','IM-1049-0002.dcm','IM-1132-0001.dcm','IM-1128-0001.dcm','IM-1062-0001-0002.dcm','IM-1057-0002.dcm','IM-1056-0002.dcm','IM-1323-0002.dcm','IM-1273-0003.dcm','IM-1452-0002.dcm','IM-1448-0001-0005.dcm','IM-1360-0002.dcm','IM-1332-0002.dcm','IM-1597-0001.dcm','IM-1594-0002.dcm','IM-1588-0002.dcm','IM-1555-0002.dcm','IM-1540-0002.dcm','IM-1523-0001.dcm','IM-1509-0001.dcm','IM-1736-0002.dcm','IM-1725-0002.dcm','IM-1670-0002.dcm','IM-1649-0003.dcm','IM-1850-0004.dcm','IM-1843-0001.dcm','IM-1808-0001.dcm','IM-1806-0006.dcm','IM-1805-0005.dcm','IM-1804-0002.dcm','IM-1792-0001.dcm','IM-1792-0001.dcm','IM-2036-0004.dcm','IM-2629-0001.dcm','IM-2611-0005.dcm','IM-3329-0002.dcm','IM-6098-0001.dcm','IM-5964-0005.dcm','IM-6216-0001.dcm','IM-6169-0002.dcm','IM-6791-0001.dcm','IM-7533-0003.dcm','IM-7511-0011.dcm','IM-7508-0008.dcm','IM-7507-0007.dcm','IM-7506-0006.dcm','IM-7505-0005.dcm','IM-7347-0001.dcm','IM-7838-0002.dcm','IM-7803-0002.dcm','IM-8738-0002.dcm','IM-8527-0002.dcm','IMG-0328-0001-0004.dcm','IMG-0324-0001.dcm','IMG-0307-0001-0006.dcm','IMG-0176-0001.dcm','IMG-0502-0002.dcm','IMT-0127-0001.dcm','IMT-0113-0002.dcm','IMT-0044-0001.dcm','IMT-0267-0002.dcm','IMT-0159-0001.dcm','IMT-0155-0003.dcm','IMT-0143-0003.dcm','IMT-0139-0001.dcm','IMT-0512-0003.dcm','IMT-0404-0001.dcm','IMT-0404-0001.dcm','IMT-0558-0002.dcm','IMT-0547-0001.dcm','IMT-0916-0001.dcm','IMT-0899-0001.dcm','IMT-0793-0004.dcm','IMT-0776-0002.dcm','IMT-0731-0001.dcm','IMT-0702-0001-0003.dcm','IMT-0961-0002.dcm','SO-0123-0002.dcm','SO-0117-0001-0002.dcm','SO-0229-0001-0002.dcm','SO-0161-0003.dcm','SO-0146-0002.dcm','SO0322-0002.dcm','SO-0302-0002.dcm','SO-0626-0002.dcm','SO-0607-0002.dcm']


for file in list:
    condition = df['FileName']==f'{file}'
    df.loc[condition, 'Projection'] = 'DV'
    print(f'changed value for {file}')


# Save the modified DataFrame
df.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_correct_DV&LL.csv', index=False)
