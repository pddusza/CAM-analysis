import pandas as pd


df = pd.read_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_correct_DV&LL.csv')
print(df)

# Function to modify the "this" part
def modify_location(location):
    return location.replace('DATA-STOR', 'OLD-DATA-STOR', 1)

# Apply the function to the 'Location' column
df['Path'] = df['Path'].apply(modify_location)

# Print the modified DataFrame
print(df)

# Save the modified DataFrame
df.to_csv('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Results/DataFrame_correct_DV&LL+COrrect_PATH.csv', index=False)