# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:55:09 2024

@author: victorsobrino
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# Load the CSV data
path = 'RestaurantNetwork/'
csv_file = 'filtered_reviews_with_communities.csv'
data = pd.read_csv(csv_file)

# Create a graph where business_id are nodes, and user_id links them
G = nx.Graph()

#%%

# Function to load the sentiment scores from the Data_Set_S1.txt file
def load_labmt(file_path):
    labmt_scores = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[4:]:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word = parts[0]
                happiness_score = parts[2]
                try:
                    labmt_scores[word] = float(happiness_score)
                except ValueError:
                    continue
    return labmt_scores

# Function to calculate sentiment score for each review
def calculate_sentiment(tokens, labmt_scores):
    total_score = 0
    count = 0
    for token in tokens:
        if token in labmt_scores:
            total_score += labmt_scores[token]
            count += 1
    return total_score / count if count > 0 else None

# Load sentiment scores from the Data_Set_S1.txt file
print('Loading sentiment scores.....\n')
labmt_scores = load_labmt("Data_Set_S1.txt")

print('Applying the sentiment calculation.....\n')
# Apply the sentiment calculation to the reviews in the data
data['tokens'] = data['text_'].str.lower().str.split()
data['sentiment_score'] = data['tokens'].apply(lambda x: calculate_sentiment(x, labmt_scores))

# Some rows have nans on the sentiment score, so let's remove those
data = data.dropna(subset=['sentiment_score'])

# Calculate sentiment statistics
sentiment_values = data['sentiment_score'].dropna().values
mean_sentiment = np.mean(sentiment_values)
median_sentiment = np.median(sentiment_values)
variance_sentiment = np.var(sentiment_values)
percentile_25 = np.percentile(sentiment_values, 25)
percentile_75 = np.percentile(sentiment_values, 75)

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(sentiment_values, bins=30, edgecolor='black', color='lightsteelblue')
plt.axvline(mean_sentiment, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {mean_sentiment:.2f}')
plt.axvline(median_sentiment, color='orange', linestyle='dashed', linewidth=1, label=f'Median: {median_sentiment:.2f}')
plt.axvline(percentile_25, color='green', linestyle='dashed', linewidth=1, label=f'25th Percentile: {percentile_25:.2f}')
plt.axvline(percentile_75, color='red', linestyle='dashed', linewidth=1, label=f'75th Percentile: {percentile_75:.2f}')
plt.legend()
plt.title("Histogram of Sentiment Scores for Restaurant Reviews", fontweight='bold')
plt.xlabel("Sentiment Score", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')
plt.show()

#%% Top/Bottom 10 restaurants

grouped_data = data[['business_id', 'sentiment_score']].groupby('business_id').mean()
merged_data = grouped_data.merge(data[['business_id', 'name']].drop_duplicates(), on='business_id')
sorted_data = merged_data.sort_values('sentiment_score', ascending=False)

# Display the top 10 happiest and saddest restaurants
print("\nTop 10 Happiest Restaurants:")
print(sorted_data[['name', 'sentiment_score']].head(10))

print("\nTop 10 Saddest Restaurants:")
print(sorted_data[['name', 'sentiment_score']].tail(10))


"""
We have to take into account that the highest value for sentiment (top 1) is 8.5 instead of 10 and the lowest is 1.3 instead of 0. AVERAGE VALUES
Rank 1      --> laughter    8.5
Rank 10222  --> terrorist   1.30
"""
#%% Categories

# Split the categories into separate rows
data_expanded = data.assign(categories=data['categories'].str.split(', ')).explode('categories')

# Group by category and calculate count, mean, and standard deviation of sentiment_score
category_sentiment = data_expanded.groupby('categories').agg(
    count=('categories', 'size'),
    sentiment_score=('sentiment_score', 'mean'),
    sentiment_score_sd=('sentiment_score', 'std'),
    sentiment_score_min=('sentiment_score', 'min'),
    sentiment_score_max=('sentiment_score', 'max')
).reset_index()

#%% Categories

# Split the categories into separate rows
data_expanded = data.assign(categories=data['categories'].str.split(', ')).explode('categories')

# Group by category and calculate count, mean, and standard deviation of sentiment_score
category_sentiment = data_expanded.groupby('categories').agg(
    count=('business_id', 'nunique'),
    sentiment_score=('sentiment_score', 'mean'),
    sentiment_score_sd=('sentiment_score', 'std')
    # sentiment_score_min=('sentiment_score', 'min'),
    # sentiment_score_max=('sentiment_score', 'max')
).reset_index()

#%% plot to find optimal threshold

# Set the threshold range (you can modify this as needed)
threshold_range = range(1, 1000)

# List to store the number of categories above each threshold
num_categories = []

# Loop through thresholds and count categories that meet the condition
for threshold in threshold_range:
    # Filter the categories with count >= threshold
    filtered_categories = category_sentiment[category_sentiment['count'] >= threshold]
    
    # Append the number of categories to the list
    num_categories.append(len(filtered_categories))

# Plot the number of categories vs threshold
plt.figure(figsize=(10, 6))
plt.plot(threshold_range, num_categories, marker='o', linestyle='-', color='purple')
plt.title("Number of Categories Above Threshold vs Threshold Value", fontweight='bold')
plt.xlabel("Threshold (Minimum Number of Reviews)", fontweight='bold')
plt.ylabel("Number of Categories", fontweight='bold')
plt.grid(True)
plt.show()

#%% Top/Bottom 10 with that threshold

threshold = 200
filtered_categories = category_sentiment[category_sentiment['count'] >= threshold]
sorted_data = filtered_categories.groupby('categories').mean().sort_values('sentiment_score', ascending=False)

print("\nTop 10 Happiest Restaurants:")
print(sorted_data.head(10))

print("\nTop 10 Saddest Restaurants:")
print(sorted_data.tail(10))

#%% LKouvain analysis

# Scatter plot for sentiment score and louvain_community
plt.figure(figsize=(10, 6))
plt.scatter(data['louvain_community'], data['sentiment_score'], alpha=0.6, color='mediumseagreen', edgecolor='black')
plt.title("Sentiment Score vs Louvain Community", fontweight='bold')
plt.xlabel("Louvain Community", fontweight='bold')
plt.ylabel("Sentiment Score", fontweight='bold')
plt.show()

# Calculate the correlation between sentiment score and louvain_community
correlation = data[['louvain_community', 'sentiment_score']].corr().iloc[0, 1]
print(f"Correlation between sentiment score and louvain community: {correlation:.2f}")











#%%

# with open(path + filtered_raw_file, 'rb') as file:
#     G = pickle.load(file)

# # Get basic information about the graph
# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()

# print(f"Number of nodes: {num_nodes}")
# print(f"Number of edges: {num_edges}")


# # Calculate degree for each node
# degrees = [deg for _, deg in G.degree()]

# # Plot the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(degrees, bins=50, edgecolor='black', alpha=0.75)
# plt.title("Degree Distribution")
# plt.xlabel("Degree")
# plt.ylabel("Frequency")
# # plt.yscale('log')  # Use logarithmic scale for better visibility
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.show()