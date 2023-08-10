import pandas as pd

def Multilabel_Binary_DataFrame(df):
    #print(df['TAG'].value_counts())
    #df=df[["FileName","TAG"]].copy()
    #df=pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP.csv')
    df=df.replace('no_finding ','no_finding')
    df=df.replace('interstitial_pattern ','interstitial_pattern')
    df=df.replace('megaesophagu','megaesophagus')
    #print(df)
    #()"this is in the fileset the list below is to be determined, based on the count"

    list_tag = ('no_finding','cardiomegaly','interstitial_pattern','alveolar_pattern','bronchial_pattern','tube','mass','pleural_effusion','pneumothorax','megaesophagus')

    #for name in list_tag:
    #    df.insert(2,name, '')

    df['TAG'].fillna('', inplace=True)

    # Create new columns for each tag
    for tag in list_tag:
        df[tag] = df['TAG'].apply(lambda x: 1 if tag in str(x).split(',') else 0)

    # Drop the original TAG column
    #df.drop('TAG', axis=1, inplace=True)
    #df.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Python/DataFrame_LLdogs_for_DP.csv', index=False)
    return df
