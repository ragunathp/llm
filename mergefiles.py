import pandas as pd

# Load the first Excel file
excel_file1 = 'file1.xlsx'
df1 = pd.read_excel(excel_file1)

# Load the second Excel file
excel_file2 = 'file2.xlsx'
df2 = pd.read_excel(excel_file2)

# Specify the common column name
common_column = 'common_column_name'

# Perform the join based on the common column
merged_df = pd.merge(df1, df2, on=common_column, how='inner')

# Save the merged DataFrame to a new Excel file
merged_excel_file = 'merged_file.xlsx'
merged_df.to_excel(merged_excel_file, index=False)

print("Merged Excel file saved successfully!")
