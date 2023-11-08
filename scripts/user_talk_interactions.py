import pandas as pd
import gzip
from tqdm import tqdm
import re

if __name__ == "__main__":
    print("Creating admin set")
    with gzip.open('../data/wiki-RfA.txt.gz', 'rt', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')  # Assuming each record is separated by a blank line

    data = []

    # Parse each block of text into a dictionary
    for block in blocks:
        record = {}
        for line in block.split('\n'):
            if line:
                key, value = line.split(':', 1)  # Split on the first colon only
                record[key.strip()] = value.strip()
        data.append(record)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    #Rename the columns
    df.columns = ['source', 'target', 'vote', 'result', 'year', 'date', 'comment']

    #Convert the year, the vote and the result to numeric values
    df['year'] = df['year'].astype(int)
    df['vote'] = df['vote'].astype(int)
    df['result'] = df['result'].astype(int)

    admin_set = set(df['source'].to_list() + df['target'].to_list())

    # Initialize a list to hold all entries
    interactions = []
    # Open the file and read in the entries
    with open("../data/enwiki-20080103.user_talk", 'r') as file:
        entry = {}
        for line in tqdm(file, desc="Filtering interactions and saving file"):
            line = line.strip()
            if line:  # If the line is not empty
                key, _, value = line.partition(' ')  # Partition line at the first space
                entry[key] = value
            else:  # If the line is empty, we've reached the end of an entry
                source = entry['REVISION'].split()[2]
                target = entry['REVISION'].split()[4]
                if "ip" not in target:
                    source = re.search(':(.*)', source).group(1)
                    if "/" in source:
                        source = source[:source.index("/")]
                    
                    if source in admin_set and target in admin_set:
                        interactions.append([source, target])

    pd.DataFrame(interactions, columns=['user1', 'user2']).to_csv("../data/interactions.csv", compression="zip")
    print("Done")
