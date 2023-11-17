# ada-2023-project-thedatafreaks
=======
# A more democratic Wikipedia
*TheDataFreaks: Robin Jaccard, Jeremy Di Dio, Daniel Tavares, Anne-Laure Tettoni, Romain Berquet*

## Abstract

Only a small fraction of wikipedia users vote in the 'request for adminship' (RfA) elections, although admins hold the key to maintaining the integrity and functionality of the world's largest online encyclopedia. From 2003 to 2013, there is an average of 53 votes per election out of the millions of registered users [1] that are allowed to vote on Wikipedia from 2003 to 2013. This low number raise some concerns for the platform's democratic processes and its robustness. A small number of voters may not accurately represent the diverse opinions and perspectives within the Wikipedia community.

Our aim is to investigate the factors influencing participation and identify the challenges that arise from the low engagement rate. Based on our research, we intend to propose solutions to increase the participation rate. 

We will study the reasons that push people to go out of their way to vote, whether it is to support someone they know, to support a community they are a part of, to support someone that share the same interests or to reject someone from an opposing group. 


## Research questions

Starting from the wikipedia request for adminship votes dataset, we discover that at the beginning, very few people voted in elections, resulting in people being elected by a minority of users. This changes over the years (there are more voters as time passes), but they still represent a very small fraction of wikipedia registered users. From this, we see that there is little incentive to vote in these elections. This renders the voting process potentially undemocratic, as a small group of (badly intentioned?) users could have a huge influence on an election, and decide to support or undermine a specific candidate, without them being able to do anything about it.



- Why do people participate in elections ? <br>
- What are the relationships between voters, if there are any ? <br>
    - Are there sub-communities within the larger community, and do these sub-communities exert any influence on the voting outcomes ? <br>
    - Is it possible to observe greater backing from individuals who have similar interests to yours ? <br>
- How could the wikipedia democratic process be improved ? <br>
- These are the questions that we will try to answer during our project.


## Additional datasets

In our research, we have incorporated two additional datasets to enrich our analysis, both accessible at https://snap.stanford.edu/data/wiki-meta.html. 

The first dataset captures edits made on user talk pages. Using this dataset, we model the interactions between users, allowing us to create communities based on these user-to-user interactions.

The second dataset that contain all edits across all Wikipedia pages. It serves as a valuable resource for understanding the editing activities of users present in our primary dataset. By looking at the modifications made by these users on various Wikipedia pages, we aim to gain insights into the content areas and topics that interest each individual.

Both dataset being extremely large, we have decided to filter them "on the fly" so that we only keep the data that concern the users that have taken part in the elections between 2003 and 2013. After this, the size of the datasets is reasonable and can be handled without problem.


## Methods

### Pre-proccessing and dataset construction

We start with the exploratory data analysis of the dataset that contains the votes in the RfA elections. We filter out some incomplete/erroneous records and perform a complete exploration of all the data by computing various descriptive statistics. 

From the dataset with all the edits of the personal user pages, we construct an unweighted and undirected graph. The nodes represent the users that are present in the RfA elections dataset and the edges connecting two users indicate that they have interacted with each other on their personal pages. 

Lastly, we use the dataset with all edits to construct a list of edited pages for each users present in the RfA elections dataset.

### Communities and interactions

We use Louvain algorithm to find communities in the graph. Our goal is to find evidence that users vote more frequently when the election concern other users of their community. For this we compute the number of intra-community votes for each community and divide this results by the expected value for the number of votes in this same community. This gives us a ratio that is significantly bigger than 1 for most communities which indicates a tendency for users to vote inside their community.


For the edits, we compute the mean of the Jaccard similarity in the lists of edited pages between every pairs of users compute the correlation with a binary value that indicate that the pair is connected by a vote. The positive correlation indicates that we are more likely to vote for (or against) someone that share similar interests as us.


We also look at the most edited pages in each topic to improve the characterization of each communities and potentially make sense of suspicious voting patterns between communities.


## Timeline
Below is a proposed timeline.

- 17.11.23: **Project milestone 2 deadline**
    - Pre-proccesing and cleaning of data finished, inital study of the communities and their influence on the votes.
---
- 18.11.23: Project work paused in order to work on Homework 2.
- 1.12.23: **Homework 2 deadline**.
---
- 8.12.23: Finish all the analysis and complete the notebook with the feedbacks received.
- 12.12.23: Decide the structure of the data story and draft it.
- 15.12.23: Have a clear plan of the presentation of the website with all the data visualizations.
- 15.12.22 - 20.12.23: Work on the data story and visualization.
- 20.12.23: Finish data story, update README.
- 22.12.23: **Project milestone 3 deadline**.

## Organization within the team
Jeremy : Pre-processing of the large datasets + graph creation<br>
Robin : Analysis of the edits of each community + proposed ideas<br>
Romain : Exploratory Data Analysis<br>
Daniel : Analysis of the interactions between all communities<br>
Anne-Laure : Writing data story + proposed ideas

## Sources


[[1] Wikimedia, number of registered users 2003-2013](https://stats.wikimedia.org/#/en.wikipedia.org/contributing/new-registered-users/normal|bar|2003-01-28~2013-05-01|~total|monthly)<br>
[Complete Wikipedia edit history](http://snap.stanford.edu/data/wiki-meta.html)<br>
[Wikipedia Requests for Adminship](http://snap.stanford.edu/data/wiki-RfA.html)<br>
[Talk pages](https://en.wikipedia.org/wiki/Help:Talk_pages)<br>
