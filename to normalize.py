import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# To read my csv with my last weights achieved after training
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\OneDrive_2024-11-25\Assignment Code\ce889_dataCollection.csv"

# Loading it 
data = pd.read_csv(file_path, header=None)

# Assigning the column order
data.columns = ['x1', 'x2', 'y1', 'y2']

# Applying Min-Max normalization
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Saving the new normalized file to work with
output_file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\OneDrive_2024-11-25\Assignment Code\normalized_ce889_dataCollection.csv"
normalized_data.to_csv(output_file_path, index=False)

print(f"Normalized data has been saved to {output_file_path}")
