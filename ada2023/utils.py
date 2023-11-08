import pandas as pd
import networkx as nx

def create_interaction_graph():
    interaction_df = pd.read_csv("../data/interactions.csv", index_col=0, compression='zip')
    interaction_df = interaction_df.groupby(['user1', 'user2']).size().reset_index(name='weight')
    G = nx.Graph()

    # Add edges and weights to the graph from the DataFrame.
    for idx, row in interaction_df.iterrows():
        G.add_edge(row['user1'], row['user2'], weight=row['weight'])
    return G
