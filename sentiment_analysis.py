# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:55:09 2024

@author: victorsobrino
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from cuisines_scrap import cuisine_scrap
import folium

# Load the CSV data
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

#%% latex table

for i in range(10):
    print(f"{i+1} & {sorted_data.iloc[i].name} & {sorted_data['sentiment_score'].iloc[i].round(3)} & {i+1} & {sorted_data.iloc[-(i+1)].name} & {sorted_data['sentiment_score'].iloc[-(i+1)].round(3)} \\\ \hline")
    # print(f"{i+1} & \multicolumn{{1}}{{l|}}{ {sorted_data.iloc[i].name} } & {sorted_data['sentiment_score'].iloc[i].round(3)} & {i+1} & \multicolumn{{1}}{{l|}}{ {sorted_data.iloc[-(i+1)].name} } & {sorted_data['sentiment_score'].iloc[-(i+1)].round(3)} \\\ \hline")


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

cuisines_wiki = cuisine_scrap()
distinct_categories = list(data_expanded['categories'].unique())

count = 0
for category in distinct_categories:
    if category.lower() in cuisines_wiki or category.lower() + ' cuisine' in cuisines_wiki:
        count += 1
        
print(f'Cuisines inside the wiki: {count}/{len(distinct_categories)}')


#%% Postal code

grouped_data = data[['postal_code', 'sentiment_score']].groupby('postal_code').mean()
sorted_data = grouped_data.sort_values('sentiment_score', ascending=False)

# Display the top 10 happiest and saddest restaurants
print("\nTop 10 Happiest Restaurants:")
print(sorted_data['sentiment_score'].head(10))

print("\nTop 10 Saddest Restaurants:")
print(sorted_data['sentiment_score'].tail(10))

#%%

import matplotlib

# Group by 'postal_code' and calculate the average sentiment score
grouped_data = data[['postal_code', 'sentiment_score']].groupby('postal_code').mean()

# Merge with original data to get 'latitude', 'longitude' for each postal_code
# We assume the latitude and longitude are the same for each postal code
grouped_with_coords = grouped_data.merge(
    data[['postal_code', 'latitude', 'longitude']].drop_duplicates(), 
    on='postal_code'
)

# Normalize the sentiment scores between 0 and 1 for color mapping
min_score = grouped_with_coords['sentiment_score'].min()
max_score = grouped_with_coords['sentiment_score'].max()
grouped_with_coords['normalized_score'] = (grouped_with_coords['sentiment_score'] - min_score) / (max_score - min_score)

# Function to map the normalized sentiment score to a color (green to red)
cmap = matplotlib.cm.get_cmap('RdYlGn')  # Red to Green colormap

def sentiment_to_color(normalized_score):
    # Convert normalized score to RGB and then to hex
    color = cmap(normalized_score)
    return matplotlib.colors.rgb2hex(color[:3])

# Create a map centered at an average location (mean of latitudes and longitudes)
mean_latitude = data['latitude'].mean()
mean_longitude = data['longitude'].mean()
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=2)

# Add circle markers for each postal code with color based on average sentiment score
for _, row in grouped_with_coords.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,  # Set a suitable radius for visibility
        color=sentiment_to_color(row['normalized_score']),
        fill=True,
        fill_color=sentiment_to_color(row['normalized_score']),
        fill_opacity=0.7,  # Adjust opacity for better visibility
        popup=f"Postal Code: {row['postal_code']}\nAverage Sentiment Score: {row['sentiment_score']:.2f}"
    ).add_to(m)

# Save the map to an HTML file and display it
m.save('postal_code_sentiment_map_simple.html')
m



#%%
import matplotlib

# Group by 'business_id' and calculate the average sentiment score
grouped_data = data[['business_id', 'sentiment_score']].groupby('business_id').mean()

# Merge with original data to get 'latitude', 'longitude', and 'name'
grouped_with_coords = grouped_data.merge(
    data[['business_id', 'latitude', 'longitude', 'name']].drop_duplicates(), 
    on='business_id'
)

# Normalize the sentiment scores between 0 and 1 for color mapping
min_score = grouped_with_coords['sentiment_score'].min()
max_score = grouped_with_coords['sentiment_score'].max()
grouped_with_coords['normalized_score'] = (grouped_with_coords['sentiment_score'] - min_score) / (max_score - min_score)

# Function to map the normalized sentiment score to a color (green to red)
cmap = matplotlib.cm.get_cmap('RdYlGn')  # Red to Green colormap

def sentiment_to_color(normalized_score):
    # Convert normalized score to RGB and then to hex
    color = cmap(normalized_score)
    return matplotlib.colors.rgb2hex(color[:3])

# Create a map centered at an average location (mean of latitudes and longitudes)
mean_latitude = data['latitude'].mean()
mean_longitude = data['longitude'].mean()
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=2)

# Add markers for each restaurant with color based on sentiment score
for _, row in grouped_with_coords.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Name: {row['name']}\nSentiment Score: {row['sentiment_score']:.2f}",
        icon=folium.Icon(color='white', icon_color=sentiment_to_color(row['normalized_score']), icon='info-sign')
    ).add_to(m)

# Save the map to an HTML file and display it
m.save('all_restaurants_sentiment_map.html')
m



#%%


# Group by postal_code and calculate mean sentiment_score
grouped_data = data[['postal_code', 'sentiment_score']].groupby('postal_code').mean()
sorted_data = grouped_data.sort_values('sentiment_score', ascending=False)

# Get the top 10 happiest and saddest postal codes
happiest = sorted_data.head(10)
saddest = sorted_data.tail(10)

# Merge to get latitude and longitude for the happiest and saddest
happiest_with_coords = happiest.merge(data[['postal_code', 'latitude', 'longitude']].drop_duplicates(), on='postal_code')
saddest_with_coords = saddest.merge(data[['postal_code', 'latitude', 'longitude']].drop_duplicates(), on='postal_code')

# Create a map centered at an average location (mean of latitudes and longitudes)
mean_latitude = data['latitude'].mean()
mean_longitude = data['longitude'].mean()
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=2)

# Add markers for the top 10 happiest restaurants
for _, row in happiest_with_coords.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Postal Code: {row['postal_code']}\nSentiment Score: {row['sentiment_score']:.2f}",
        icon=folium.Icon(color='green')
    ).add_to(m)

# Add markers for the top 10 saddest restaurants
for _, row in saddest_with_coords.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Postal Code: {row['postal_code']}\nSentiment Score: {row['sentiment_score']:.2f}",
        icon=folium.Icon(color='red')
    ).add_to(m)

# Save the map to an HTML file and display it
m.save('happiest_saddest_restaurants_map.html')
m


