import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle

# Set the file path to your dataset
file_path = 'precovid_reviews.csv'

# Initialize an empty graph
G = nx.Graph()

# Loop through the file in chunks
for chunk in tqdm(pd.read_csv(file_path, usecols=['name', 'user_id'])):
    # Iterate over each user group in the chunk
    for user_id, group in chunk.groupby('user_id'):
        # Get the list of business names that this user reviewed
        businesses = group['name'].tolist()

        # Connect each business (restaurant) that the user has reviewed to every other business they reviewed
        for i in range(len(businesses)):
            for j in range(i + 1, len(businesses)):
                business_a = businesses[i]
                business_b = businesses[j]
                
                # Add an edge between the two businesses
                G.add_edge(business_a, business_b)

# Get the number of nodes (restaurants) and edges (connections between restaurants)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Print the results
print(f"Number of restaurants (nodes): {num_nodes}")
print(f"Number of connections (edges): {num_edges}")

# Additional: View some example nodes and edges to understand the structure
print("\nSample of nodes (restaurants):", list(G.nodes)[:10])  # First 10 nodes
print("\nSample of edges (connections):", list(G.edges)[:10])  # First 10 edges


# Saving the graph using pickle
with open("user_restaurant_graph_no_reviews.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph has been saved successfully.")
