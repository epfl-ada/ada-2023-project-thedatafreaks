# Bob’s journey to adminship: Exploring the world of Wikipedia Requests for Adminship

*TheDataFreaks: Robin Jaccard, Jeremy Di Dio, Daniel Tavares, Anne-Laure Tettoni, Romain Berquet*

## Datastory

Embark on a captivating exploration of Bob's journey towards admission by following the link to our insightful [data narrative](https://dioday45.github.io/TheDataFreaks/).

## Abstract

Only a small fraction of wikipedia users vote in the 'request for adminship' (RfA) elections, although admins hold the key to maintaining the integrity and functionality of the world's largest online encyclopedia. From 2003 to 2013, there is an average of 53 votes per election out of the millions of registered users [1] that are allowed to vote. This low number raises some concerns for the platform's democratic processes and its robustness. A small number of voters may not accurately represent the diverse opinions and perspectives within the Wikipedia community.

Our aim is to investigate the factors influencing participation, identify challenges that arise from the low engagement rate, and examine carefully the fairness of the elections. Based on our research, we aim to propose effective solutions to enhance the fairness and overall participation rate.

## Research questions

Starting from the wikipedia request for adminship votes dataset, we discover that very few people vote in elections, resulting in people being elected by a minority of users. From this, we see that there is little incentive to vote in these elections. This renders the voting process potentially undemocratic, as a small group of badly intentioned users could have a huge influence on an election by supporting or undermining a specific candidate.

Is the election process as fair as it should ideally be?<br>
What drives people to vote?<br>
How could the Wikipedia democratic process be improved?<br>


These are the questions that we will try to answer during our project.


## Additional datasets

In our research, we have incorporated two additional datasets to enrich our analysis, both accessible at https://snap.stanford.edu/data/wiki-meta.html. 

The first dataset captures edits made on user talk pages. Using this dataset, we model the interactions between users, allowing us to create communities based on these user-to-user interactions.

The second dataset that contains all edits across all Wikipedia pages. It serves as a valuable resource for understanding the editing activities of users present in our primary dataset. By looking at the modifications made by these users on various Wikipedia pages, we aim to gain insights into the content areas and topics that interest each individual.

Both dataset being extremely large, we have decided to filter them "on the fly" so that we only keep the data that concern the users that have taken part in the elections between 2003 and 2013. After this, the sizes of the datasets are reasonable and can be handled without problem.

## Methods

### Pre-processing and dataset construction

We start with the exploratory data analysis of the dataset that contains the votes in the RfA elections. We filter out some incomplete/erroneous records and perform a complete exploration of all the data by computing various descriptive statistics. 

From the dataset with all the edits of the personal user pages, we construct a graph. The nodes represent the users that are present in the RfA elections dataset and the edges connecting two users indicate that they have interacted with each other on their personal pages. The graph is weighted (by the number of interactions) and undirected. We do not include joint edits on the interaction graph. This has the advantage of only displaying direct and conscious interactions. The results are independent of the edits which makes any potential connection between clusters and shared interests more relevant. 

Lastly, we use the dataset with all edits to construct a list of edited pages for each user present in the RfA elections dataset.

### Individual perspectives

To understand what drives people to vote, we create a balanced dataset by pairing (for a given election) a voter and a non-voter that have similar voting statistics. Then we conduct a logistic regression, incorporating two key features: the number of contacts from the voter who voted before him and a binary variable representing contact between the voter and the person running for election. 

Additionally, we conduct a logistic regression to predict the value of the vote using two different features. The initial pair of features corresponds to the number of contacts who voted positively for the election before our user and the number of contacts who voted negatively before him. The second feature is a binary feature that represents that a  communication exists between the candidate and the voter. 


### Communities and interactions

We use the Louvain algorithm to find communities in the graph. To make a bit more sense of those communities, we look at the most edited pages in each community to characterize each community and potentially make sense of suspicious voting patterns.

To find evidence that the communities play a role in influencing the presence of votes between two users, we compute the numbers of votes from each community to every other community and normalize these values by the corresponding expected numbers of votes based on community sizes. This gives us ratios that can be compared to 1. Indeed if the communities have no influence we should not expect to see ratios significantly bigger or smaller than 1.

We also analyze vote results to determine the percentage of positive votes between the different communities. After filtering out the non-significant results, we assessed support by calculating the difference in positive votes percentage from the baseline for each community pair. For the baseline, we use the percentage of positive votes over the whole dataset. With those results, we create a graph that illustrates support or opposition between communities.

For the edits, we compute the mean of the Jaccard similarity in the lists of edited pages between every pair of users and compute the correlation with a binary value that indicates that the pair is connected by a vote. The positive correlation indicates if we are more likely to vote for (or against) someone that shares similar interests as us.

To identify the presence of fake accounts, we filter the users of the dataset based on their votes and edits. If a user creates multiple accounts to artificially inflate positive votes, it's probable that they won't invest time in making edits with each account before casting a vote.
To identify elections possibly influenced by a large number of fake accounts, we compute the ratio of such accounts (meeting both criteria simultaneously) participating in each election and inspect more carefully the elections with the highest ratios.


## Code organization

In the ```notebooks/main_notebook.ipynb```, you can find all the detailed analyses that are presented in the datastory.  

In the ```ada2023``` folder, you can find the main logic to build the graph.

The ```scripts``` folder contains the logic to filter the data on the fly.

Here is an overview of the codebase:



```
.
├── README.md
├── ada2023
│   ├── __init__.py
│   └── utils.py
├── data
│   ├── graph_communities.png
│   ├── interactions.csv
│   ├── interactions_edits_grouped.zip
│   ├── wiki-RfA-cleaned.csv
│   └── wiki-RfA.txt.gz
├── notebooks
│   └── main_notebook.ipynb
├── requirements.txt
└── scripts
    ├── user_edit_interactions.py
    └── user_talk_interactions.py
```

## Organization within the team
Jeremy : Pre-processing of the large datasets and website creation <br>
Robin : Analysis of the edits of each community, visualization plots, and data story + <br>
Romain : Exploratory Data Analysis and analysis of RfA at the individual level <br>
Daniel : Analysis of the interactions between all communities and visualization plots <br>
Anne-Laure : Data story writing, illustration creation, and community analysis

## Sources

[Wikimedia, number of registered users 2003-2013](https://stats.wikimedia.org/#/en.wikipedia.org/contributing/new-registered-users/normal|bar|2003-01-28~2013-05-01|~total|monthly)<br>
[Complete Wikipedia edit history](http://snap.stanford.edu/data/wiki-meta.html)<br>
[Wikipedia Requests for Adminship](http://snap.stanford.edu/data/wiki-RfA.html)<br>
[Talk pages](https://en.wikipedia.org/wiki/Help:Talk_pages)<br>
[Paper of Cabunducan, G.](https://ieeexplore.ieee.org/document/5992657)
