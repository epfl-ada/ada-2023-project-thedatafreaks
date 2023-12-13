# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:ada] *
#     language: python
#     name: conda-env-ada-py
# ---

# %% [markdown]
# # _Main Notebook_

# %% [markdown]
# ## Table of Contents:
#
# - [EDA](#eda)
#   - [Data Loading](#eda_data)
#   - [Preliminary checks](#eda_checks)
#   - [Voting results analysis](#eda_results)
#   - [Number of votes analysis](#eda_analysis)
# - [Communities analysis](#communities)
#   - [Setup](#communities_setup)
#   - [Interaction Graph](#communites_interaction)
#   - [Communities](#communities_communities)
#   - [Vote Analysis](#communities_vote)
# - [Content of edits analysis](#edits)
#   - [Setup](#edits_setup)
#   - [Statistics](#edits_statistics)
#   - [Investigation of most edited pages](#edits_investigation)

# %%
# Imports
import pandas as pd
import sys
sys.path.append('../')
from ada2023.utils import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gzip
from itertools import combinations
from scipy.stats import pearsonr
from statsmodels.stats import diagnostic
from scipy import stats
import statsmodels.formula.api as smf

# %% [markdown]
# # Exploratory Data Analysis <a class="anchor" id="eda"></a>

# %% [markdown]
# ### Data Loading <a class="anchor" id="eda_data"></a>

# %%
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
df.columns = ['source', 'target', 'vote', 'result', 'year_election', 'date_vote', 'comment']

# %%
df

# %% [markdown]
# ### Preliminary checks <a class="anchor" id="eda_checks"></a>

# %%
#Create a new dataframe before cleaning the data
new_df = df.copy(deep=True)

# %% [markdown]
# ##### 1 - Dive into user name source

# %%
#Transform the source column to string
source_cleaned_data = new_df.copy(deep=True)
source_cleaned_data['source'] = source_cleaned_data['source'].astype(str)

# %%
#Look at the distribution of the length of the source tags with a box plot
ax = source_cleaned_data['source'].str.len().plot(kind='box', patch_artist=True, 
                                                  boxprops=dict(facecolor='skyblue'))
ax.set_title('Distribution of Length of Source Tags', fontsize=14)
ax.set_ylabel('Length of Source Tags', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
#Look at the number of unique source users
unique_voters = source_cleaned_data['source'].nunique()
print(f'The number of unique voters is {unique_voters}')

#Look at the number of null values for the source
nan_source = source_cleaned_data[source_cleaned_data.source == '']['source'].count()
print(f'The number of voters without tags is {nan_source}')

# %% [markdown]
# While examining the outliers in relation to their source tags:
#
# It's observed that outliers possessing source tags longer than 25 characters typically do not present specific issues.
# A significant portion of these outliers are identified to have empty source tag lengths. Consequently, we've opted to exclude votes linked with empty source tags. This decision aligns with our objective to utilize the data for community building and to track user interactions. Allowing votes from empty source tags might skew our community analysis, potentially leading to an imbalance where certain users' votes are disproportionately influential compared to others.

# %%
#Remove the rows with votes associated to empty source 
source_cleaned_data = source_cleaned_data[source_cleaned_data.source != '']

# %%
#Now we look at the other outliers, votes with user tags of length greater than 200
source_cleaned_data[source_cleaned_data.source.str.len() > 20].source.unique()

# %% [markdown]
# Usernames appear accurate and suitable for user tags, and thus do not require removal.
#

# %% [markdown]
# ##### 2 - Dive into target user name

# %%
#Make deep copy before cleaning for target
target_cleaned_data = source_cleaned_data.copy(deep=True)

# %%
#Look at the distribution of the length of the target tags with a box plot
ax = target_cleaned_data['target'].str.len().plot(kind='box', patch_artist=True, 
                                                  boxprops=dict(facecolor='skyblue'))
ax.set_title('Distribution of Length of Target Tags', fontsize=14)
ax.set_ylabel('Length of Source Tags', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
#Look at the number of unique target users
unique_electives = target_cleaned_data['target'].nunique()
print(f'The number of unique users running for election is {unique_electives}')

#Look at the number of null values for the source
nan_target = target_cleaned_data[target_cleaned_data.target == '']['target'].count()
print(f'The number of nan values for the target is {nan_target}')

# %%
new_df[new_df.target.str.len() > 20].target.unique()

# %% [markdown]
# Usernames appear accurate and suitable for user tags, and thus do not require removal.

# %% [markdown]
# ##### 3 - Check the date and time of votes :

# %%
date_cleaned_data = target_cleaned_data.copy(deep=True)


# %%
# Define a function to extract date components
def extract_date_components(date_str):
    try:
        # Split the date string by the comma and space to separate time and date parts
        time_part, date_part = date_str.split(', ')
        # Split the time part by the colon to separate hours and minutes
        hour, minute = time_part.split(':')
        # Split the date part by space to separate day, month, and year
        day, month, year = date_part.split(' ')
        
        return pd.Series({
            "hour": hour,
            "minute": minute,
            "day": day,
            "month": month,
            "year_vote": year
        })
    except ValueError:
        # If there is a ValueError, return None for each component
        return pd.Series({
            "hour": None,
            "minute": None,
            "day": None,
            "month": None,
            "year_vote": None
        })

# Apply the function to each row in the 'date' column
date_components = date_cleaned_data['date_vote'].apply(extract_date_components)

# Concatenate the new DataFrame with the original one (if needed)
date_cleaned_data = pd.concat([date_cleaned_data, date_components], axis=1)

date_cleaned_data

# %%
#Look at the proportion of rows with missing date_vote
non_date_votes = date_cleaned_data[date_cleaned_data.date_vote == ''].date_vote.count()
total_count = date_cleaned_data.date_vote.count()
print(f'The number of votes for which the date is missing or incorrect is {non_date_votes}')
print(f'This represents {(non_date_votes/total_count)*100:.2f}% of the data.')

# %%
#Remove the rows with missing date_vote
date_cleaned_data = date_cleaned_data[date_cleaned_data.date_vote != '']

# %%
#Look at the distribution for the values of the hour with histogram
date_cleaned_data['hour'].value_counts().sort_index().plot(kind='bar' , color = 'teal' ,  edgecolor='black')
plt.xlabel('Hour the vote was cast')
plt.ylabel('Number of votes cast')
# Set a grid for easier reference to the quantities
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Distribution of votes by hour of day')
plt.show()

# %%
#Proportion of rows with the hour value as 31
ratio_of_31 = date_cleaned_data[date_cleaned_data.hour == "31"]["hour"].count()/date_cleaned_data["hour"].count()
print(f'The proportion of rows with the hour value as 31 is {ratio_of_31}')

# %% [markdown]
# In this dataset, there are a small fraction of votes occurring at the 31st hour, which is not a valid time. Given that the number of occurrences is negligible, we have chosen to exclude this data point from the dataframe. This removal is unlikely to affect the overall analysis of the dataset due to its minimal incidence.

# %%
#Remove from the dataframe the rows with the value of the hour as 31
date_cleaned_data = date_cleaned_data[date_cleaned_data['hour'] != '31']

# %%
# Set the figure size for better visibility
plt.figure(figsize=(15, 10))

ax = date_cleaned_data['minute'].value_counts().sort_index().plot(kind='bar', color='teal', edgecolor='black')

ax.set_xlabel('Minute of the hour', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of values for the minutes', fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %% [markdown]
# The minute values appear to be in order, and their distribution is evenly spread, which aligns with expectations.
#
#

# %%
#Look at the distribution of the values for the days with histogram and order the values

ax = date_cleaned_data['day'].value_counts().sort_index().plot(kind='bar' ,  color='teal', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Day the vote was cast', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of values for the days', fontsize=16)

plt.show()

# %% [markdown]
# The values for the day also seem to be correct.
#
#

# %%
#Look at the distribution of the values for the months with histogram and order the values
ax = date_cleaned_data['month'].value_counts().sort_index().plot(kind='bar' , color='teal', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Month the vote was cast', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of values for the months', fontsize=16)

plt.show()

# %% [markdown]
# The dataset displays variations in the representation of specific months. For instance, the month of July is listed as 'Jul,' 'Julu,' and 'July'; similarly, October is noted as 'Oct' and 'October.'

# %%
#Map the values of the months to the full name of the month
month_map = { 
    "Apr" : "April",
    "April" : "April",
    "Aug" : "August",
    "August" : "August",
    "Dec" : "December",
    "December" : "December",
    "Feb" : "February",
    "February" : "February",
    "Jan" : "January",
    "Janry" : "January",
    "January" : "January",
    "Jul" : "July",
    "Julu" : "July",
    "July" : "July",
    "Jun" : "June",
    "June" : "June",
    "Mar" : "March",
    "March" : "March",
    "May" : "May",
    "Mya" : "May",
    "Nov" : "November",
    "November" : "November",
    "Oct" : "October",
    "October" : "October",
    "Sep" : "September",
    "September" : "September"
}

def correction_month (month) : 
    return month_map.get(month, month)

date_cleaned_data['month'] = date_cleaned_data['month'].apply(correction_month)
date_cleaned_data

# %% [markdown]
# The values for the years seems also to be ok.
#
#

# %%
date_cleaned_data['date_vote'] = pd.to_datetime(date_cleaned_data['day'].astype(str) + ' ' +
                            date_cleaned_data['month'].astype(str) + ' ' +
                            date_cleaned_data['year_vote'].astype(str) + ' ' +
                            date_cleaned_data['hour'].astype(str) + ':' +
                            date_cleaned_data['minute'].astype(str),
                            format='%d %B %Y %H:%M' , errors = 'coerce')

date_cleaned_data.drop(['hour', 'minute', 'day', 'month', 'year_vote'], axis=1, inplace=True)
date_cleaned_data

# %% [markdown]
# ##### 4 - Dive into the year_election values

# %%
year_elections_cleaned_data = date_cleaned_data.copy(deep=True)

# %%
#Look at the distribution of the values for year_election
year_elections_cleaned_data['year_election'] = year_elections_cleaned_data['year_election'].astype(int)


ax = year_elections_cleaned_data['year_election'].value_counts().sort_index().plot(kind='bar' , color='teal', 
                                                                                   edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Year the election took place', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of the values for year_election', fontsize=16)

plt.show()

# %% [markdown]
# ##### 5 - Dive into the vote and results values
#

# %%
vote_results_data_cleaned = year_elections_cleaned_data.copy(deep=True)

# %%
#Look at the distribution of the values for the vote
vote_results_data_cleaned['vote'] = vote_results_data_cleaned['vote'].astype(int)
vote_results_data_cleaned['result'] = vote_results_data_cleaned['result'].astype(int)

print(vote_results_data_cleaned['vote'].describe())
ax = vote_results_data_cleaned['vote'].value_counts().sort_index().plot(kind='bar' , color='teal', 
                                                                        edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Vote', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of the values for the votes', fontsize=16)

plt.show()

# %%
value_perc_vote = vote_results_data_cleaned['vote'].value_counts(normalize=True) * 100

# Print the percentages
print("Percentage of Each Unique Value in vote:")
print(value_perc_vote)

# %%
#Look at the distribution of the values for the result
print(vote_results_data_cleaned['result'].describe())
ax = vote_results_data_cleaned['result'].value_counts().sort_index().plot(kind='bar' , color='teal', 
                                                                          edgecolor='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Election result', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)
ax.set_title('Distribution of the values for the results of the elections', fontsize=16)

plt.show()
plt.show()

# %%
value_perc_result = vote_results_data_cleaned['result'].value_counts(normalize=True) * 100

# Print the percentages
print("Percentage of Each Unique Value in result:")
print(value_perc_result)

# %% [markdown]
# ##### 6 - Dive into comments

# %%
#Look at the proportion of empty comments
nb_empty_com = vote_results_data_cleaned[vote_results_data_cleaned.comment == ""]["comment"].count()
ratio_empty_com = nb_empty_com/vote_results_data_cleaned["comment"].count()
print(f'The percentage of empty comments is {ratio_empty_com:.2%}%')

#Look at the disribution of the length of the comments
ax = vote_results_data_cleaned['comment'].str.len().plot(kind='box', patch_artist=True, 
                                                         boxprops=dict(facecolor='skyblue'))
ax.set_title('Distribution of Length of Comments', fontsize=14)
ax.set_ylabel('Length of Comments', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
cleaned_df = vote_results_data_cleaned.copy(deep=True)

#Store the cleaned dataframe in a csv file
cleaned_df.to_csv('../data/wiki-RfA-cleaned.csv', index=False)

# %% [markdown]
# ### Voting results analysis <a class="anchor" id="eda_results"></a>

# %%
#Import the cleaned dataframe
analysis_df = pd.read_csv('../data/wiki-RfA-cleaned.csv')

# %%
analysis_df['date_vote'] = pd.to_datetime(analysis_df['date_vote'])

# %% [markdown]
# ##### 1 - User behavior

# %%
#First we plot the distribution of the number of votes per user
grouped_per_user = analysis_df.groupby('source').apply(lambda x : pd.Series({
    'number_of_votes' : len(x['target'])})).reset_index()

plt.figure(figsize=(12, 6))

ax = sns.histplot(grouped_per_user['number_of_votes'], color='teal', log=True, bins=1000, edgecolor='black')

ax.set_title('Distribution of Number of Votes', fontsize=16)  
ax.set_xlabel('Number of Votes', fontsize=14)  
ax.set_ylabel('Frequency (Log Scale)', fontsize=14) 
ax.grid(True) 
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

# %% [markdown]
# We can see a classic long-tail distribution of voter activity, indicative of a pattern where a small number of individuals account for a disproportionately large number of votes, while the vast majority participate minimally. The steep decline and subsequent long tail to the right suggest that the community has a few highly engaged users, a common trait in voluntary, community-driven platforms. This could imply that engagement initiatives might focus on the more active users to leverage their influence, or conversely, on the less active majority to increase overall participation.

# %%
date_analysis = analysis_df.groupby('source').apply(lambda x : pd.Series({
    'sequence_of_votes' : x['date_vote'].values})).reset_index()

def calculation_duration (dates) : 
     
    sorted_dates = sorted(dates)
    return (sorted_dates[-1] - sorted_dates[0])

date_analysis['duration'] = date_analysis['sequence_of_votes'].apply(calculation_duration)

#Look into the distribution of the duration of the sequence of votes
plt.figure(figsize=(12, 6))

ax = sns.histplot(date_analysis['duration'].dt.days, color='teal', log =True,bins = 1000,  edgecolor='black')

ax.set_title('Distribution of User Voting Duration', fontsize=15, pad=20)
ax.set_xlabel('Duration in Days', fontsize=12)
ax.set_ylabel('Frequency (Log Scale)', fontsize=12)
ax.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()


# %% [markdown]
# The histogram depicting the duration between users' first and last votes confirms the conclusion form the previous plot : most users engage in a short burst of activity, casting votes for a brief period before becoming inactive, as shown by the numerous tall bars at the plot's start. This trend aligns with the initial surge of participation seen in the previous plot, where many users voted only a few times. Conversely, the long tail in both plots points to a subset of dedicated users who not only vote more frequently but also stay active over long stretches, suggesting a core group's persistent engagement shapes the platform's voting landscape. Together, these insights reveal a pattern of engagement where a small cohort of users provides ponctual votes and others who have a really important impact.

# %%
#Now to get an inuition about the average number of election that a user runs for
#As we know that an RfA is on average 7 days, we will set this specific time lapse as the duration of an election

def estimate_number_of_elections(group):
    # Ensure the dates are sorted
    sorted_dates = group.sort_values()
    
    # Initialize the count of elections and set the first election start date
    elections_count = 1
    election_start = sorted_dates.iloc[0]

    for date in sorted_dates[1:]:
        # If the current vote is more than a week after the election start, count a new election
        if (date - election_start).days > 7:
            elections_count += 1
            election_start = date

    return elections_count

estimated_number_elections_per_user = (analysis_df.groupby('target')['date_vote']
                                                  .apply(estimate_number_of_elections)
                                                  .reset_index()
                                                  .rename(columns={'date_vote': 'estimated_number_elections'}))

#We look at the distribution of values for the estimated number of elections
print(estimated_number_elections_per_user.describe())

#We look at the distribution of values for the estimated number of elections
sns.histplot(estimated_number_elections_per_user.estimated_number_elections, bins=30, kde=False, 
             color='skyblue', log = True)
plt.title('Distribution of Estimated Number of Elections Per Target')
plt.xlabel('Estimated Number of Elections')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# The histogram and descriptive statistics indicate that the majority of targets in the dataset are estimated to have run for just one election, which is consistent with the data's mean and mode both being close to 1. The significant drop-off after one election suggests that it is much less common for targets to run multiple times. With over 75% of the targets estimated to have participated in only one election, there's a clear indication that repeated runs for election are rare. This aligns with the earlier plots showing that while there are a few highly active and engaged individuals, the vast majority have lower levels of activity. In this context, the repeated candidacies could be attributed to a similarly small group of individuals who are highly involved in the community, repeatedly seeking election, reflecting a pattern of engagement where a core group of users is significantly more active than the rest.

# %%
# Get the users who ran for election only once
ran_once = (estimated_number_elections_per_user[estimated_number_elections_per_user
                                                .estimated_number_elections == 1])

# Get the users who ran for election only once and did not get elected
not_elected_once = ran_once[~ran_once.target.isin(analysis_df[analysis_df.result == 1].target)]

number_unique_people_running = analysis_df.target.nunique()
ratio_give_up_after_one = len(not_elected_once)/number_unique_people_running
print(f'The proportion of users that give up after one election is {ratio_give_up_after_one:.2%}%')

# %% [markdown]
# The statistic that 39% of voters do not participate in subsequent elections if they don't succeed in their first attempt is indicative of a high dropout rate, potentially reflecting various underlying factors. This rate suggests a significant level of voter apathy or disillusionment, possibly arising from a perception that their vote lacks impact or from a general lack of sustained motivation. This statistic also poses questions about the evolving nature of voter engagement and the effectiveness of current electoral systems in sustaining voter interest and belief in the electoral process.

# %% [markdown]
# ##### 2 - Election dynamics

# %%
elect_dynamics_df = analysis_df.copy(deep=True)

# %%
# We set an id fo each of the election following the method used before in order to compute 
# further statistics regarding the elections

# Sort the dataframe by 'target' and 'date_vote'
elect_dynamics_df.sort_values(by=['target', 'date_vote'], inplace=True)

# Initialize a counter for the global election ID
global_election_id = 0
# Initialize the last seen election date for each target
last_election_date = elect_dynamics_df.groupby('target')['date_vote'].first() - pd.Timedelta(days=8)

# Function to assign election ids
def assign_election_ids(row):
    global global_election_id
    # If the current vote date is more than 7 days after the last election date for this target
    if (row['date_vote'] - last_election_date[row['target']]).days > 7:
        global_election_id += 1
        last_election_date[row['target']] = row['date_vote']
    return global_election_id

# Apply the function to each row
elect_dynamics_df['global_election_id'] = elect_dynamics_df.apply(assign_election_ids, axis=1)

# %%
elect_dynamics_df


# %%
#We compute here different statistics
elect_dynamics_df = elect_dynamics_df.groupby(['global_election_id']).apply(lambda x : pd.Series({
    'number_of_votes' : len(x['source']), 
    'ratio_positive_votes' : x[x.vote == 1]['vote'].sum() / len(x.source), 
    'ratio_neutral_votes' : x[x.vote == 0]['vote'].sum() / len(x.source),
    'average_comment_length' : x['comment'].str.len().mean(),
    'date_last_vote' : x['date_vote'].max(),
    'result' : x['result'].max(),
    'year_election' : x['year_election'].max(),
})).reset_index()
elect_dynamics_df

# %%
plt.figure(figsize=(15, 10))  

sns.scatterplot(x='date_last_vote',
             y='ratio_positive_votes', 
             hue='result', 
             style='result', 
             data=elect_dynamics_df,
             palette='Set2') 

plt.title('Trends in Ratio of Positive Votes by Election Outcome Over Time', fontsize=18)
plt.xlabel('Date of Vote', fontsize=16)
plt.ylabel('Ratio of Positive Votes', fontsize=16)
plt.xticks(rotation=45, fontsize=14)  
plt.yticks(fontsize=14)
plt.legend(title='Election Outcome', fontsize=14, title_fontsize=16)
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()


# %% [markdown]
# The scatter plot illustrates a correlation between the ratio of positive votes and election outcomes, with a dense cluster of green dots at higher ratios indicating wins and red dots at lower ratios indicating losses. As the number of votes increases, there seems to be a trend toward more wins, shown by the prevalence of green dots in areas with a greater number of votes. Elections with a moderate ratio of positive votes show a mix of outcomes, reflecting the competitive nature of those elections. Overall, the plot suggests that while a higher number of votes is generally favorable, the ratio of positive votes is a strong indicator of success in elections, as most wins are concentrated in the region with higher positive vote ratios
#
# Seems like there is a specific threshold for the percentage of positive votes for an election to be successfull.

# %%
plt.figure(figsize=(15, 10))  

sns.scatterplot(x='date_last_vote',
             y='number_of_votes', 
             hue='result', 
             style='result', 
             data=elect_dynamics_df,
             palette='Set2', 
             alpha = 0.6) 

plt.title('Trends in Ratio of Number of votes by Election Outcome Over Time', fontsize=18)
plt.xlabel('Date of Vote', fontsize=16)
plt.ylabel('Number of votes', fontsize=16)
plt.xticks(rotation=45, fontsize=14)  
plt.yticks(fontsize=14)
plt.legend(title='Election Outcome', fontsize=14, title_fontsize=16)
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()

# %% [markdown]
# The plot suggests that the -1 outcome is commonly associated with a lower number of votes, while the 1 outcome shows greater variability with several outliers indicating exceptionally high vote counts. Despite the presence of both outcomes throughout the time range, there's no apparent temporal trend in voting patterns. We will further analyse this below by loooking at each year and try to detect wether we have further insights. 
#
# We will use below statistical measure to determine the real relation between the different variables.

# %%
#We compute the correlation between the ratio of positive votes and the outcome of the election
stats.pearsonr(elect_dynamics_df['ratio_positive_votes'], elect_dynamics_df['result'])

# %%
#We compute the correlation between the number of votes and the outcome of the election
stats.pearsonr(elect_dynamics_df['number_of_votes'], elect_dynamics_df['result'])

# %%
regression_df = elect_dynamics_df.copy(deep=True)
regression_df['result'] = regression_df['result'].replace({-1 : 0})
mod = smf.logit(formula='result ~  (year_election) + number_of_votes + ratio_positive_votes + \
                          + ratio_positive_votes + average_comment_length' , data=regression_df)
res = mod.fit()
print(res.summary())

# %% [markdown]
# Model Fit: The R-squared value is 0.8318, which is high, suggesting that the model fits the data well.
#
# Significance: The LLR (likelihood ratio test) p-value is less than 0.05, indicating that the model as a whole is statistically significant compared to the null model.
#
# Regarding the different coefficients :
#
# The ratio_positive_votes coefficient is significant (p < 0.05) and positive, indicating that as the ratio of positive votes increases, the log-odds of winning the election (result=1) significantly increase.
# The year_election, number_of_votes, and average_comment_length coefficients are not statistically significant (p > 0.05), implying that these variables do not have a significant impact on the log-odds of the election outcome in the presence of other variables.
# Intercept: The intercept is also not significant, which is not typically a concern as it simply sets the baseline log-odds of the outcome when all predictors are at zero.
#
# In summary, the model strongly suggests that the ratio of positive votes is a key predictor of election outcomes, while other variables like the year of the election, the number of votes, and the average comment length do not show a significant relationship in this logistic regression model. The presence of quasi-separation suggests that while the model fits the current data well, it might not generalize well to new data.

# %% [markdown]
# ### Number of votes analysis <a class="anchor" id="eda_analysis"></a>

# %%
elect_dynamics_df['year_election'].unique().sort()

# %%
#Plot the distribution of the number of elections per year
ax = elect_dynamics_df['year_election'].value_counts().sort_index().plot(kind='bar' , 
                                                                         color='teal', edgecolor='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xlabel('Year the election took place', fontsize=14)
ax.set_ylabel('Number of occurrences', fontsize=14)

plt.show()

# %% [markdown]
# We clearly have an imbalance number of elections through the years. We will analyse the trend over time of votting patterns, wether in specific years we had more elections with positiv outcome. 

# %%
# Assuming 'election_year' is of type int
for year in sorted(elect_dynamics_df['year_election'].unique()):
    data_subset = elect_dynamics_df[elect_dynamics_df['year_election'] == year]
    
    sns.histplot(x='number_of_votes', data=data_subset, hue='result', log_scale=(False, False), 
                 color = 'skyblue', edgecolor='black' , palette= 'Set2')
    plt.title(f'Histogram for Election Year {int(year)}')
    plt.show()

# %%
sns.histplot(x = 'number_of_votes', data = elect_dynamics_df , hue = 'result', log_scale= (True, False))
plt.show()

# %%
print(new_df.groupby(["target", "result"])["source"].count().median())
print(new_df.groupby(["target", "result"])["source"].count().mean())

# %% [markdown]
# We have a specific pattern that is quite common every years regarding the elections. We have a high proportion of elections that have a low number of votes and those elections most of the time end with a bad outcome (election being -1). The election that end with a positive outcome tend to rely on an important number of votes.

# %% [markdown]
# # Communities Analysis <a class="anchor" id="communities"></a>

# %% [markdown]
# ### Setup <a class="anchor" id="communities_setup"></a>

# %% [markdown]
# We process the Wikipedia Request for Adminship (RfA) dataset into a dataframe. We are using a [Wikipedia edit history dataset](https://snap.stanford.edu/data/wiki-meta.html) containing edit up to january 2008. Therefore we filter votes that aren't present in this timeframe.

# %%
df = pd.read_csv("../data/wiki-RfA-cleaned.csv")

# We filter out all the votations after 2008 as we do not have the edits for those dates
df = df[df.year_election < 2009]

#Set of users that are present in the adminship dataset
admin_set = set(df['source'].to_list() + df['target'].to_list())

# %%
print(f"Number of users present in the adminship dataset : {len(admin_set)}")

# %% [markdown]
# ### Interaction Graph <a class="anchor" id="communities_interaction"></a>

# %% [markdown]
# We consider an interaction between two users to be an edit from user A in the user talk page of user B. User talk page ["normal use is for messages from, and discussion with, other editors"](https://en.wikipedia.org/wiki/Wikipedia:User_pages). We filtered edits to keep only interactions from users that where present in the RfA dataset. Using those interactions we created an undirected graph where the weight is the number of interaction between the two users and each node is a user.

# %%
G = create_interaction_graph()

# %%
print(f"Number of nodes (users) in the graph : {len(G)}")
print(f"Number of users in the RfA dataset : {len(admin_set)}")
print(f"Percentage of users in the graph : {(len(G)/len(admin_set)):.2%}")

# %%
interactions_df = df.copy(deep=True)
interactions_df['vote'] = interactions_df['vote'].astype(int)

def get_interaction_weight(graph, node1, node2):
    try:
        # Retrieve the weight attribute of the edge between node1 and node2
        weight = graph[node1][node2].get('weight', 0)
    except KeyError:
        # If there's no edge between node1 and node2, the interaction count is 0
        weight = 0
    return weight

# Apply the function to each row of the DataFrame
interactions_df['interaction_count'] = interactions_df.apply(lambda row: get_interaction_weight(G, row['source'], row['target']), axis=1)

interactions_df

# %%
#Look at the distribution of the number of interactions per election
plt.figure(figsize=(15, 10))

ax = sns.histplot(interactions_df['interaction_count'], color='teal', log=True, bins=1000, edgecolor='black')

ax.set_title('Distribution of Number of Interactions', fontsize=16)
ax.set_xlabel('Number of Interactions', fontsize=14)
ax.set_ylabel('Frequency (Log Scale)', fontsize=14)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()

# %% [markdown]
# We will explore the influence of user interactions, as captured in our interaction graph, on their participation and choices in each other's voting processes. Specifically, we aim to analyze whether interactions between users impact both the likelihood of participating in one another's elections and the eventual voting decisions.

# %%
edge_list = []

for u, v, data in G.edges(data=True):
    weight = data.get('weight', 0)
    edge_list.append({'source': u, 'target': v, 'weight': weight})
    edge_list.append({'source': v, 'target': u, 'weight': weight})  # Add reverse direction

edges_df = pd.DataFrame(edge_list)

edges_df

# %%
# Create different filters based on the weight column of edges_df
weight_thresholds = np.linspace(5, 60, num= 12, endpoint=True)
mean_ratios = []

for threshold in weight_thresholds:
    # Filter edges_df based on the current threshold
    filtered_edges_df = edges_df[edges_df['weight'] >= threshold]
    
    # Recompute the grouped_interactions DataFrame with the new filter
    filtered_grouped_interactions = filtered_edges_df.groupby('target').apply(
        lambda x: pd.Series({
            'number_of_distinct_interactions': filtered_edges_df[filtered_edges_df['target'] == x.name]['source'].nunique(),
            'number_interactors_voting': sum(filtered_edges_df[filtered_edges_df['target'] == x.name]['source'].isin(interactions_df[interactions_df['target'] == x.name]['source']))
        })
    ).reset_index()
    
    filtered_grouped_interactions['ratio_interactors_voting'] = filtered_grouped_interactions['number_interactors_voting'] / filtered_grouped_interactions['number_of_distinct_interactions']
    mean_ratio = filtered_grouped_interactions['ratio_interactors_voting'].mean()
    mean_ratios.append(mean_ratio)

# Plotting
plt.plot(weight_thresholds, mean_ratios, marker='o')
plt.xlabel('Minimum Interaction Weight Threshold')
plt.ylabel('Mean Ratio of Interactors Voting')
plt.title('Mean Ratio of Interactors Voting by Interaction Weight Threshold')
plt.grid(True)
plt.show()

# %% [markdown]
# There is a sharp increase in the mean ratio of interactors voting for the target as the minimum interaction weight threshold increases from the lowest value up to a certain point. This suggests that as you consider only those pairs with more significant interactions (a higher weight), there is a higher likelihood that they will participate in each other's elections. This part of the trend indicates a strong positive relationship between interaction intensity and voting participation.
# The increasing trend in the ratio of voters plateaus, which may imply that beyond a certain point, increasing the threshold for interaction weight does not significantly influence the likelihood of users participating in each other's votes. It can be inferred that there might be a saturation point beyond which the strength of interaction (as quantified by weight) does not have much additional impact on voting participation.

# %%
effect_comm_df = interactions_df.groupby('interaction_count').apply(lambda x : pd.Series({
    'percentage_positive_votes' : x[x.vote == 1]['vote'].sum() / len(x.source), 
    'number_of_votes' : len(x.source),
    'proportion_of_voters' : len(x.source)/len(df['source'].unique())
})).reset_index()

effect_comm_df = effect_comm_df[effect_comm_df['number_of_votes'] >= 25]

plt.figure(figsize=(15, 10))
ax = sns.lineplot(x='interaction_count',
                  y='percentage_positive_votes',
                  data=effect_comm_df,
                  color='teal',
                  marker='o',
                  markersize=10,
                  markerfacecolor='skyblue',
                  markeredgecolor='black',
                  markeredgewidth=1)

ax.set_title('Percentage of Positive Votes by Number of Interactions', fontsize=20, pad=20)
ax.set_xlabel('Number of Interactions', fontsize=15)
ax.set_ylabel('Percentage of Positive Votes', fontsize=15)
ax.grid(True)  
ax.set_xticks(effect_comm_df['interaction_count']) 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# %% [markdown]
# Despite the variability, there seems to be a general trend where the percentage of positive votes tends to increase with the number of interactions, especially noticeable in the initial section of the plot. However, this trend is not consistent across the entire range of interaction counts.

# %% [markdown]
# To have a better understanding of the interactions, we plot them in a graph. We also plot the degree rank plot and histogram. The degree of a node is the number of edges adjacents to the node. This plot helps us to better understand the distribution of the number of adjacent nodes. We can see that most of the nodes have a low degree.

# %%
# We sort the nodes in the graph by their degree
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
unique_degree, counts = np.unique(degree_sequence, return_counts=True)

# Degree histogram
plt.bar(unique_degree, counts,width=10, color='b')
plt.title("Degree histogram")
plt.xlabel("Degree")
plt.ylabel("# of Nodes")

plt.show()

# %% [markdown]
# ### Communities <a class="anchor" id="communities_communities"></a>

# %% [markdown]
# We explore the relationship between users by creating communities. Communities are created using Louvain algorithm that "[works in 2 steps. On the first step it assigns every node to be in its own community and then for each node it tries to find the maximum positive modularity gain by moving each node to all of its neighbor communities. If no positive gain is achieved the node remains in its original community](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)"

# %%
#We create the communities

communities = nx.community.louvain_communities(G, resolution=1.5, seed=2)

# %%
print(f"Number of communities in graph of users with interactions : {len(communities)}")

# %%
for i, c in enumerate(communities):
    print(f"Community {i} has size {len(c)}")

# %% [markdown]
# ### Vote analysis <a class="anchor" id="communities_vote"></a>

# %% [markdown]
# To understand the influence of communities, we compute the probability of vote to be within your community if it was voted at random.

# %%
n = len(G) # Number of nodes in the graph
p_same_cluster = 0 # Probability that a random vote is an intra-cluster vote
array_p_same_cluster = np.array([])

# We compute the probability that a random vote is an intra-cluster vote
for c in communities:
    p_same_cluster += (len(c)/n)*((len(c)-1)/n)
    array_p_same_cluster = np.append(array_p_same_cluster, (len(c)/n)*((len(c)-1)/n))
print(f"Probability that a random vote is an intra-cluster vote in interaction graph : {p_same_cluster:.2%}")

# %%
# Initialize a counter for votes within the same community
intra_vote_count = np.zeros(len(communities))

# Iterate through the dataframe
for index, row in df.iterrows():
    entity1 = row['source']
    entity2 = row['target']

    # Check if entities are in the same community
    for count, community in enumerate(communities):
        if entity1 in community and entity2 in community:
            intra_vote_count[count] += 1

# Print the result
print(f"Number of votes within the same community : {int(intra_vote_count.sum())}")

# %%
# Initialize a counter for votes in the graph
votes_in_the_graph = 0

# Iterate through the dataframe
for index, row in df.iterrows():
    entity1 = row['source']
    entity2 = row['target']

    # Check if entities are in the graph
    if entity1 in G and entity2 in G:
        votes_in_the_graph += 1

# Print the result
print(f"Number of votes in the graph : {votes_in_the_graph}")

# %% [markdown]
# We compare the probability of a random vote with what was observed. The goal is to assess whether the users vote is influenced by its community. We observe that we have a ~3x increase in probability to vote towards your own community.

# %%
print(f"Effective percentage of intra-cluster votes in G: {(intra_vote_count.sum()/votes_in_the_graph):.2%}")

# %% [markdown]
# To understand the increase in votes, we compute the expected number of votes if voted at random. Then we make the ratio to derive the multiplicative coefficient from the expected number of votes to observed. 

# %%
# expected number of intra-cluster votes
expected_nb_votes = array_p_same_cluster * votes_in_the_graph

# %%
# ratio of effective intra-cluster votes over expected intra-cluster votes
vote_gain = intra_vote_count / expected_nb_votes
vote_gain

# %% [markdown]
# We want to understand the distribution of votes between communities. For that we plot the distribution of votes between communities

# %%
# Number of votes across communities
vote_count_matrix = np.zeros((len(communities), len(communities)))
nb_community_votes = np.zeros(len(communities))

# %%
# Populate the vote count matrix
for index, row in df.iterrows():
    entity1 = row['source']
    entity2 = row['target']
    if entity1 in G and entity2 in G:
        i_src = find_community(entity1, communities)
        i_dst = find_community(entity2, communities)
        vote_count_matrix[i_src][i_dst] += 1
        nb_community_votes[i_src] += 1

# %%
# We verify our computations and transform the vote count matrix into a ratio matrix
np.testing.assert_array_equal(vote_count_matrix.sum(axis=1), nb_community_votes)
ratio_vote_count_matrix = (vote_count_matrix / nb_community_votes[:, np.newaxis])*100 
np.testing.assert_almost_equal(ratio_vote_count_matrix.sum(axis=1), np.ones(len(communities))*100)

# %% [markdown]
# In the plot we observe that the destination communities that recieve most of the votes are the larger communities. This is explained by the fact that for large communities, more votation take place and therefore more votes are directed to them.

# %%
# Heatmap of the ratio of votes across communities
plt.figure(figsize=(24, 12))
sns.heatmap(ratio_vote_count_matrix, cmap="Blues", annot=True, fmt=".1f", linewidths=.5, linecolor="black")
plt.title("Percentage of votes per communities in G")
plt.xlabel("Destination community")
plt.ylabel("Source community")
plt.show()

# %% [markdown]
# To mitigate this domination of the large communities, we scale the results by the probability of a vote between two communities given that the votes are random. We display the ratio of effective votes over expected votes for each pair of communities.

# %%
# Create matrix that represents the probability of a vote between two communities
prob_vote_community_matrix = np.zeros((len(communities), len(communities)))
for i_src in range(len(communities)):
    for i_dst in range(len(communities)):
        prob_vote_community_matrix[i_src][i_dst] = (len(communities[i_src])*len(communities[i_dst]))/(len(G)**2)
# create matrix that represents the expected number of votes between two communities
ratio_vote_expected_matrix = prob_vote_community_matrix * votes_in_the_graph

# %%
# populate the matrix of votes
vote_result_matrix = [[np.zeros(3) for i in range(len(communities))] for j in range(len(communities))]
nb_result_votes = np.zeros((len(communities), len(communities)))
for index, row in df.iterrows():
    entity1 = row['source']
    entity2 = row['target']
    if entity1 in G and entity2 in G:
        i_src = find_community(entity1, communities)
        i_dst = find_community(entity2, communities)
        if row['vote'] == 1:
            vote_result_matrix[i_src][i_dst][2] += 1
        elif row['vote'] == -1:
            vote_result_matrix[i_src][i_dst][0] += 1
        else:
            vote_result_matrix[i_src][i_dst][1] += 1
        nb_result_votes[i_src][i_dst] += 1


# %%
gain_vote_expected_matrix = np.nan_to_num(nb_result_votes / ratio_vote_expected_matrix)

# %%
# Heatmap of the gain from expected votes across communities
# Couleurs sympas: 'PuBuGn', 'RdYlBu', 'coolwarm'
plt.figure(figsize=(36, 12))
sns.heatmap(gain_vote_expected_matrix, cmap='PuBuGn', annot=True, fmt=".2f", linewidths=.5, linecolor="black")
plt.title("Multiplicative relation expected number of votes to observed number of votes per communities in G")
plt.xlabel("Destination community")
plt.ylabel("Source community")
plt.show()

# %% [markdown]
# The diagonal of the matrix has significantly higher values. This indicates that people tend to vote more for the people part of their community. We recall that the communities have been created with the interactions between the users and not the votes.

# %% [markdown]
# #### Can we find a rivalry between some communities? Maybe a community only vote negatively towards another community.

# %% [markdown]
# We want to know the voting habitudes of communities. For that we plot the result precentage per community

# %%
# populate the matrix of votes
vote_result_matrix = [[np.zeros(3) for i in range(len(communities))] for j in range(len(communities))]
nb_result_votes = np.zeros((len(communities), len(communities)))
for index, row in df.iterrows():
    entity1 = row['source']
    entity2 = row['target']
    if entity1 in G and entity2 in G:
        i_src = find_community(entity1, communities)
        i_dst = find_community(entity2, communities)
        if row['vote'] == 1:
            vote_result_matrix[i_src][i_dst][2] += 1
        elif row['vote'] == -1:
            vote_result_matrix[i_src][i_dst][0] += 1
        else:
            vote_result_matrix[i_src][i_dst][1] += 1
        nb_result_votes[i_src][i_dst] += 1


# %%
perc_result_matrix = np.nan_to_num((vote_result_matrix / nb_result_votes[:,:,np.newaxis]))*100

# %%
for_ratio_result_matrix = perc_result_matrix[:,:,2]
significance_matrix = [[[None]*3 for i in range(len(communities))] for j in range(len(communities))]
for i in range(len(communities)):
    for j in range(len(communities)):
        if int(nb_result_votes[i][j]) != 0:
            significance_matrix[i][j][0] = stats.binomtest(int(vote_result_matrix[i][j][2]), 
                                                         n=int(nb_result_votes[i][j]), 
                                                         p=value_perc_vote[1]/100,
                                                          )
plt.figure(figsize=(36, 12))
sns.heatmap(for_ratio_result_matrix, cmap="Greens", annot=True, fmt=".1f", linewidths=.5, linecolor="black")
for i in range(len(communities)):
    for j in range(len(communities)):
        if significance_matrix[i][j][0]!=None and significance_matrix[i][j][0].pvalue < 0.01:
            plt.scatter(j+0.85, i+0.35, color='black', marker='*')
plt.title("Percentage of votes \"for\" per communities in G")
plt.xlabel("Destination community")
plt.ylabel("Source community")
plt.show()

# %%
neutral_ratio_result_matrix = perc_result_matrix[:,:,1]
for i in range(len(communities)):
    for j in range(len(communities)):
        if int(nb_result_votes[i][j]) != 0:
            significance_matrix[i][j][1] = stats.binomtest(int(vote_result_matrix[i][j][1]), 
                                                         n=int(nb_result_votes[i][j]), 
                                                         p=value_perc_vote[0]/100)
        
plt.figure(figsize=(36, 12))
sns.heatmap(neutral_ratio_result_matrix, cmap="Greys", annot=True, fmt=".1f", linewidths=.5, linecolor="black")
for i in range(len(communities)):
    for j in range(len(communities)):
        if significance_matrix[i][j][1]!=None and significance_matrix[i][j][1].pvalue < 0.01:
            plt.scatter(j+0.85, i+0.35, color='black', marker='*')
plt.title("Percentage of votes \"neutral\" per communities in G")
plt.xlabel("Destination community")
plt.ylabel("Source community")
plt.show()

# %%
against_ratio_result_matrix = perc_result_matrix[:,:,0]
for i in range(len(communities)):
    for j in range(len(communities)):
        if int(nb_result_votes[i][j]) != 0:
            significance_matrix[i][j][2] = stats.binomtest(int(vote_result_matrix[i][j][0]), 
                                                         n=int(nb_result_votes[i][j]), 
                                                         p=value_perc_vote[-1]/100)
plt.figure(figsize=(36, 12))
sns.heatmap(against_ratio_result_matrix, cmap="Reds", annot=True, fmt=".1f", linewidths=.5, linecolor="black")
for i in range(len(communities)):
    for j in range(len(communities)):
        if significance_matrix[i][j][2]!=None and significance_matrix[i][j][2].pvalue < 0.01:
            plt.scatter(j+0.85, i+0.35, color='black', marker='*')
plt.title("Percentage of votes \"against\" per communities in G")
plt.xlabel("Destination community")
plt.ylabel("Source community")
plt.show()

# %%
size_plot = len(communities)
fig, ax = plt.subplots(figsize=(size_plot, size_plot), nrows=len(communities), ncols=len(communities), layout='constrained')
for i in range(len(communities)):
    for j in range(len(communities)):
        diff_for = perc_result_matrix[i][j][2] - value_perc_vote[1]
        diff_neutral = perc_result_matrix[i][j][1] - value_perc_vote[0]
        diff_against = perc_result_matrix[i][j][0] - value_perc_vote[-1]
        ax[i,j].bar(x=['for', 'neutral', 'against'], height=[diff_for, diff_neutral, diff_against], color=['green', 'grey', 'red'])
        ax[i,j].set_xticks([])
        ax[i,j].set_ylim(-70, 70)
        for k in range(3):
            if significance_matrix[i][j][k]!=None and significance_matrix[i][j][k].pvalue < 0.01:
                ax[i,j].scatter(x=[k], y=[60], marker='*', color='black')

for i, ax in enumerate(fig.get_axes()):
    if i < len(communities):
        ax.set_xlabel(f'{i%len(communities)}', size='large', fontweight='bold') 
        ax.xaxis.set_label_position('top')
    if i%len(communities) == 0 :
        ax.set_ylabel(f'{i//len(communities)}', size='large', fontweight='bold')
    else :
        ax.label_outer()
    if i//len(communities) == len(communities)-1:
        ax.set_xlabel(f'{i%len(communities)}', size='large', fontweight='bold') 
fig.suptitle("Difference between observed and mean percentage of votes per communities in G", size='x-large', fontweight='bold')
fig.supxlabel("Destination community", size='x-large', fontweight='bold')
fig.supylabel("Source community", size='x-large', fontweight='bold')
plt.show()


# %% [markdown]
# We can observe that some communities display either a positive or negative bias in their voting preferences. If votes were at random and participation uniform accross communities, we would have expected that the portion of votes "for", "against" and "neutral" to have to same proportion between communities. Suspicious results could help us to inspect further the relationship between the two communities involved.

# %%
import gravis as gv

# %%
communities_name = ["Pop Culture Mix", "Middle East & Religion", "Varied Interests", "USA Historical Figures", 
              "Australia", "Religion Debates & Controversies", "Controversial Pop Culture", 
              "Russia & Eastern Europe", "USA & east Asia mix", "New Zealand", "Military Aircraft", "Youth Pop Culture", 
              "India & South Asia", "Historical & Political mix", "People Mix", 
              "USA Varied Interest", "Science", "Historical Figures", 
              "UK & Ireland", "TV Series 'Lost'", "Sports", "Scientology", 
              "Canada & Ice Hockey", "Comics", "Balkans & Central Asia", 
              "Chemical Elements", "Wrestling", "Oregon", "Politics"]

# %%
G = nx.DiGraph()

for i, c in enumerate(communities):
    G.add_node(i)
    G.nodes[i]['size'] = len(c)
    G.nodes[i]['name'] = communities_name[i]
pos = nx.circular_layout(G, scale=700)
for i in range(len(communities)):
    G.nodes[i]['x'] = pos[i][0]
    G.nodes[i]['y'] = pos[i][1]
baseline = value_perc_vote[1]
for src in range(len(communities)):
    for dst in range(len(communities)):
        if significance_matrix[src][dst][0]!=None and significance_matrix[src][dst][0].pvalue < 0.01:
            ci = significance_matrix[src][dst][0].proportion_ci(confidence_level=0.99)
            diff_for = ci.low*100 - baseline if ci.low*100 > baseline  else baseline - ci.high*100
            G.add_edge(src,dst, weight=diff_for)
            color = "green" if ci.low > 0.7 else "red"
            nx.set_edge_attributes(G, {(src,dst):{"color":color}})

gv.vis(G,layout_algorithm_active=False, use_node_size_normalization=True, node_size_normalization_max=60, node_size_normalization_min=7, node_hover_neighborhood=True, node_label_size_factor=2.0, node_label_data_source='name', edge_size_data_source='weight', edge_size_factor=0.5)

# %% [markdown]
# # Content of edits analysis <a class="anchor" id="edits"></a>

# %% [markdown]
# This section explores the relationship between the topics of Wikipedia pages edited by users and the occurrence of votes between two users. The goal is to identify potential correlations and patterns that would show that editing similar topics has an influence in the motivation to cast a vote.

# %% [markdown]
# ### Setup <a class="anchor" id="edits_setup"></a>

# %%
# The original dataset can be found here (https://snap.stanford.edu/data/wiki-meta.html). 
# The version that we use here has already been modified so that we get each user and 
# the page they modified with the number of edits

edits_df = pd.read_csv("../data/interactions_edits_grouped.zip", index_col=0, compression='zip')
edits_df

# %%
# We create a list of Wikipedia pages modified by each users
user_indices = edits_df.groupby('username').apply(lambda x: x.index.tolist()).reset_index(name='Indices')
user_indices

# %%
# Create a list of all users present in the edits dataset
users = set(edits_df['username'].tolist())

# %% [markdown]
# Create a matrix with the Jaccard index (on the the lists of modified pages) for all pairs of users. Jaccard index "[is a statistic used for gauging the similarity and diversity of sample sets](https://en.wikipedia.org/wiki/Jaccard_index)". It will be used to understand the similarity of edited pages between pairs of users.

# %% [raw]
# # Initialization of the matrix
# matrix_similarity = pd.DataFrame(index=list(users), columns=list(users))
# matrix_similarity

# %% [raw]
# # Calculate Jaccard index between each pair of users
# # Takes a few hours to run...
# for i, user_tuple in enumerate(combinations(list(users), 2)):
#
#     pages1 = user_indices[user_indices['username'] == user_tuple[0]]['Indices'].iloc[0]
#     pages2 = user_indices[user_indices['username'] == user_tuple[1]]['Indices'].iloc[0]
#     
#     jaccard_index = jaccard_similarity(set(pages1), set(pages2))
#
#     matrix_similarity.at[user_tuple[0], user_tuple[1]] = jaccard_index
#     matrix_similarity.at[user_tuple[1], user_tuple[0]] = jaccard_index
#
# # Fill diagonal with 1 since the Jaccard index with oneself is always 1
# matrix_similarity = matrix_similarity.fillna(1.0)
#
# print(matrix)

# %% [raw]
# matrix_similarity.to_csv('jaccard.csv', index=True)

# %%
matrix_similarity = pd.read_csv("../data/jaccard.csv.zip", index_col=0, compression='zip')
matrix_similarity

# %% [markdown]
# Create a DataFrame with all pairs of users and a binary variable that indicates if a vote exists for each pair. It will be helpful to computes the correlation between similarity in edited pages and voting interaction between two users.

# %% [raw]
# similarity_and_vote = pd.DataFrame(index=list(combinations(list(users), 2)), columns=['vote', 'jaccard'])

# %% [raw]
# # Fill in the new similarity_and_vote with the values from matrix_similarity
# # Takes 30 minutes to run...
# for i, index_row in enumerate(similarity_and_vote.iterrows()):
#     similarity_and_vote.at[index_row[0], 'jaccard'] = matrix_similarity.at[index_row[0][0], index_row[0][1]]

# %% [raw]
# # Fill in the new similarity_and_vote with the binary values that indicate the presence of the votes
# list_users = list(matrix_similarity.index)
#
# for index, row in df.iterrows():
#     if (row['source'] in list_users) & (row['target'] in list_users):
#         if (row['source'], row['target']) in similarity_and_vote.index:
#             similarity_and_vote.at[(row['source'], row['target']), 'vote'] = 1
#         elif (row['target'], row['source']) in similarity_and_vote.index:
#             similarity_and_vote.at[(row['target'], row['source']), 'vote'] = 1
# similarity_and_vote = similarity_and_vote.fillna(0)

# %% [raw]
# similarity_and_vote.to_csv('jaccard_and_votes.csv', index=True)

# %%
similarity_and_vote = pd.read_csv("../data/jaccard_and_votes.csv.zip", index_col=0, compression='zip')
similarity_and_vote

# %% [markdown]
# ### Statistics <a class="anchor" id="edits_statistics"></a>

# %% [markdown]
# Now we will compute some statistics on this data.

# %%
pearsonr(similarity_and_vote['vote'], similarity_and_vote['jaccard'])

# %% [markdown]
# The correlation between similarity score on edited pages and the votes is not very strong but positive with high significance.

# %%
# Mean similarity between all pairs of users
mean_sim_all = similarity_and_vote['jaccard'].mean()
mean_sim_all

# %%
# Mean similarity between pairs of users that are linked by a vote
mean_sim_vote = similarity_and_vote[similarity_and_vote['vote'] == 1]['jaccard'].mean()
mean_sim_vote

# %%
print(f"People that are linked by a vote have {mean_sim_vote / mean_sim_all:.2f} "
      f"times more common edited pages than the average.")

# %% [markdown]
# ### Investigation of most edited pages <a class="anchor" id="edits_investigation"></a>

# %% [markdown]
# We investigate most edited pages per community. The goal is to find a common topic that could define community's interest.

# %%
for count, community in enumerate(communities):
    community_to_check = community
    
    user_list = list(user_indices['username'])
    all_subject = set()
    for user in community_to_check:
        if user in user_list:
            all_subject = (set(user_indices[user_indices['username'] == user]['Indices'].iloc[0])
                           .union(all_subject))
            
    data = pd.DataFrame(index=list(all_subject), columns=['count'])
    data = data.fillna(0)
    
    for user in community_to_check:
        if user in user_list:
            subjects = user_indices[user_indices['username'] == user]['Indices'].iloc[0]
            for s in subjects:
                data.at[s, 'count'] += 1
    print(f"\nCommunity {count}:")            
    print(data.sort_values('count', ascending = False).head(20))
