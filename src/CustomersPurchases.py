import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('data/input/customer_data.csv') 

# Set 'CustomerID' as the index for easier manipulation
df.set_index('CustomerID', inplace=True)

# Step 1: Clean the data - remove dollar signs and spaces
df.fillna(0, inplace=True)
df_cleaned = df.replace({'$': '', ' ': ''}, regex=True)  # Remove dollar signs and spaces
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, handle any non-numeric values

# print(df_cleaned.dtypes)

# Step 2: Handle missing values - Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
df_cleaned_imputed = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Step 3: Normalize the data (important for clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned_imputed)

# Step 4: Perform K-Means clustering (let's assume we want 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned_imputed['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Summarize each cluster
cluster_summary = df_cleaned_imputed.groupby('Cluster').mean()

# Step 6: Visualize the clusters (optional)
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned_imputed.index, df_cleaned_imputed['Dept 1'], c=df_cleaned_imputed['Cluster'], cmap='viridis', marker='o', edgecolor='black')
plt.title("Clustering of Customers based on Dept 1 Sales Potential")
plt.xlabel("Customer ID")
plt.ylabel("Sales in Dept 1 ($)")
plt.show()

# Display cluster summary
print("Cluster Summary (Average Sales per Department for each Cluster):")
print(cluster_summary)

# Step 7: Description of a typical customer per group
# For each cluster, describe the typical customer by their average sales in each department
for i in range(3):
    print(f"\nCluster {i} - Typical Customer:")
    typical_customer = cluster_summary.iloc[i]
    print(typical_customer)