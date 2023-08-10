import pandas as pd

def DeletePNGfromFileName(df):
    #df = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Multi_Label_dataset/DataFrame_LLdogs_for_DP.csv')
    print(df)

    # Function to modify the "this" part
    def modify_location(location):
        return location.replace('.png', '', 1)

    # Apply the function to the 'Location' column
    df['FileName'] = df['FileName'].apply(modify_location)

    # Print the modified DataFrame
    print(df)
    return df
   
