import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
# Load the customer data 
customer_df = pd.read_csv(r"C:\Users\HP\OneDrive\task4\Mall_Customers.csv")
# Display the first few rows of the dataset
print(customer_df.head()) 
# Get a summary of the dataset 
print(customer_df.info()) 
# Check for missing values 
print(customer_df.isnull().sum()) 
# Select relevant features for segmentation 
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] 
X = customer_df[features] 
# Standardize the features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
# Elbow Method to find the optimal number of clusters 
inertia = [] 
for k in range(1, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42) 
    kmeans.fit(X_scaled) 
    inertia.append(kmeans.inertia_) 
plt.figure(figsize=(8, 6)) 
plt.plot(range(1, 11), inertia, marker='o') 
plt.xlabel('Number of Clusters') 
plt.ylabel('Inertia') 
plt.title('Elbow Method for Optimal Number of Clusters') 
plt.show() 
# Train K-Means clustering model with the optimal number of clusters 
k = 5  # Example: choose 5 clusters based on the elbow method 
kmeans = KMeans(n_clusters=k, random_state=42) 
kmeans.fit(X_scaled) 
# Assign clusters to the data 
customer_df['Cluster'] = kmeans.labels_ 
# Scatter plot of clusters based on different pairs of features 
plt.figure(figsize=(12, 8)) 
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
hue='Cluster', data=customer_df, palette='viridis', s=100) 
plt.title('Customer Segmentation based on Income and Spending Score') 
plt.show()
customer_df.to_csv("customers_with_clusters.csv", index=False)
cluster_profile.to_csv("cluster_profile.csv")


