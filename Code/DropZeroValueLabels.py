import pandas as pd

def dropzerolabels(df):
    list_tag = ['FileName', 'TAG' ,'no_finding','cardiomegaly','interstitial_pattern','alveolar_pattern','bronchial_pattern','tube','mass','pleural_effusion','pneumothorax','megaesophagus']
    print(df.shape)
    df = df[df[list_tag].sum(axis=1) > 0]
    print(df.shape)
    return df
