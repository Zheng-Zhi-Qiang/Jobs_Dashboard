import re
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from IPython.display import display
from serpapi import GoogleSearch
from plotly.graph_objs import Bar, Layout
from plotly import offline
from nltk.tokenize import MWETokenizer
from google.cloud import bigquery

# Project Details
project = "trans-gate-374512"
dataset_id = "Singapore_Jobs"
dataset_ref = bigquery.DatasetReference(project, dataset_id)

# Current Date
current_date = datetime.date.today()

google_search_api_key = "2790b3e4171eab9e05796d210149d3b9db3f21e59baaac28a01edb107c4ae2a1"
search_term = "Data Analyst"
search_location = "Singapore"

params = {
    "engine": "google_jobs",
    "q": search_term,
    "api_key": google_search_api_key,
    "google_domain": "google.com",
    "location": search_location,
    "chips": "date_posted:today",
    "start": 0,
    "num": "100",
    "no_cache": "true",
}

# Keywords to look out for
keywords_programming = [
    'sql', 'python', 'r', 'c', 'c#', 'javascript', 'js',  'java', 'scala', 'sas', 'matlab', 
    'c++', 'c/c++', 'perl', 'go', 'typescript', 'bash', 'html', 'css', 'php', 'powershell', 'rust', 
    'kotlin', 'ruby',  'dart', 'assembly', 'swift', 'vba', 'lua', 'groovy', 'delphi', 'objective-c', 
    'haskell', 'elixir', 'julia', 'clojure', 'solidity', 'lisp', 'f#', 'fortran', 'erlang', 'apl', 
    'cobol', 'ocaml', 'crystal', 'javascript/typescript', 'golang', 'nosql', 'mongodb', 't-sql', 'no-sql',
    'visual_basic', 'pascal', 'mongo', 'pl/sql',  'sass', 'vb.net', 'mssql', 
]

keywords_libraries = [
    'scikit-learn', 'jupyter', 'theano', 'openCV', 'spark', 'nltk', 'mlpack', 'chainer', 'fann', 'shogun', 
    'dlib', 'mxnet', 'node.js', 'vue', 'vue.js', 'keras', 'ember.js', 'jse/jee',
]

keywords_analyst_tools = [
    'excel', 'tableau',  'word', 'powerpoint', 'looker', 'powerbi', 'outlook', 'azure', 'jira', 'twilio',  'snowflake', 
    'shell', 'linux', 'sas', 'sharepoint', 'mysql', 'visio', 'git', 'mssql', 'powerpoints', 'postgresql', 'spreadsheets',
    'seaborn', 'pandas', 'gdpr', 'spreadsheet', 'alteryx', 'github', 'postgres', 'ssis', 'numpy', 'power_bi', 'spss', 'ssrs', 
    'microstrategy',  'cognos', 'dax', 'matplotlib', 'dplyr', 'tidyr', 'ggplot2', 'plotly', 'esquisse', 'rshiny', 'mlr',
    'docker', 'linux', 'jira',  'hadoop', 'airflow', 'redis', 'graphql', 'sap', 'tensorflow', 'node', 'asp.net', 'unix',
    'jquery', 'pyspark', 'pytorch', 'gitlab', 'selenium', 'splunk', 'bitbucket', 'qlik', 'terminal', 'atlassian', 'unix/linux',
    'linux/unix', 'ubuntu', 'nuix', 'datarobot',
]

keywords_cloud_tools = [
    'aws', 'azure', 'gcp', 'snowflake', 'redshift', 'bigquery', 'aurora',
]

keywords_general_tools = [
    'microsoft', 'slack', 'apache', 'ibm', 'html5', 'datadog', 'bloomberg',  'ajax', 'persicope', 'oracle', 
]

keywords_general = [
    'coding', 'server', 'database', 'cloud', 'warehousing', 'scrum', 'devops', 'programming', 'saas', 'ci/cd', 'cicd', 
    'ml', 'data_lake', 'frontend',' front-end', 'back-end', 'backend', 'json', 'xml', 'ios', 'kanban', 'nlp',
    'iot', 'codebase', 'agile/scrum', 'agile', 'ai/ml', 'ai', 'paas', 'machine_learning', 'macros', 'iaas',
    'fullstack', 'dataops', 'scrum/agile', 'ssas', 'mlops', 'debug', 'etl', 'a/b', 'slack', 'erp', 'oop', 
    'object-oriented', 'etl/elt', 'elt', 'dashboarding', 'big-data', 'twilio', 'ui/ux', 'ux/ui', 'vlookup', 
    'crossover',  'data_lake', 'data_lakes', 'bi', 
]

# Keywords list
keywords = keywords_programming

# Bind Multi-word tokens
token_dict = {}
mwe_tokenizer = MWETokenizer(separator="_")
regx = re.compile('[,.;@#_+-?!&$/]+')
for string in keywords:
    if string.isalnum() != True:
        separator = regx.findall(string)
        if separator[0] == '#' or separator[0] == '++':
            mwe_tokenizer.add_mwe(tuple(string.split()))
        else:
            token_list = string.split(separator[0])
            mwe_tokenizer.add_mwe(tuple(token_list))
            key = re.sub('[,.;@#_+?!&$/]+', '_', string)
            token_dict[key] = string

# Execute the search to retrieve the results
# for x in range(3):
#     start = x * 10
#     params["start"] = start
#     search = GoogleSearch(params)
#     results = search.get_dict()
#     jobs = results["jobs_results"]
#     jobs_df = pd.DataFrame(jobs)
#     print(len(jobs_df.index))
#     jobs_df = pd.concat([jobs_df, pd.json_normalize(jobs_df["detected_extensions"])], axis=1).drop("detected_extensions", 1)
#     if 'salary' in jobs_df.columns:
#         jobs_df = jobs_df.drop('salary', 1)
    
#     if x == 0:
#         current_day_jobs_df = jobs_df
#     else:
#         current_day_jobs_df = current_day_jobs_df.append(jobs_df, ignore_index=True)

# # Add date column to dataframe and drop related links and extensions
# current_day_jobs_df['date'] = pd.to_datetime(current_date)
# current_day_jobs_df = current_day_jobs_df.drop('related_links', 1).drop('extensions', 1)

# # Insert data into BigQuery
client = bigquery.Client()
# table_ref = dataset_ref.table("raw_jobs")
# table = client.get_table(table_ref)
# errors = client.insert_rows_from_dataframe(table, current_day_jobs_df)
# if errors == []:
#     print(errors)
#     print('Upload failed')
# else: 
#     print('Data loaded into table')

# Extract full dataset from BigQuery
table_ref = dataset_ref.table("raw_jobs")
table = client.get_table(table_ref)

complete_jobs_df = client.list_rows(table).to_dataframe()


# Convert the description column into a list
description_list = complete_jobs_df["description"].values.tolist()

# Tokenize the strings to obtain list of tokens
description_tokens = []
for description in description_list:

    # Remove punctuations from strings
    cleaned_description = re.sub('[,.;@_?!&$/]+\s*', ' ', description)

    # Tokenize the string
    cleaned_tokens = mwe_tokenizer.tokenize(cleaned_description.lower().split())

    # Replace the Multi-word token with the actual token
    for index, string in enumerate(cleaned_tokens):
        if string in list(token_dict.keys()):
            cleaned_tokens[index] = token_dict[string]

    # Add to complete list of tokens        
    description_tokens += cleaned_tokens
    

# Filter out stopwords and words that are not in keyword lists
filtered_description_tokens = list(filter(lambda x: x in keywords, description_tokens))

# Convert result list to dataframe
description_tokens_df = pd.DataFrame({'tokens': filtered_description_tokens})

# Count the number of occurence for each token
token_counts = description_tokens_df.groupby('tokens').size().to_frame().reset_index()
token_counts.columns = ['tokens', 'count']

# Overwrite current frequency table in BigQuery
table_ref = dataset_ref.table("total_token_frequencies")
table = client.get_table(table_ref)

job_config = bigquery.job.LoadJobConfig()
job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
client.load_table_from_dataframe(token_counts, table_ref, job_config=job_config)

# Plot frequency graph of tokens
token_counts = token_counts.sort_values(by=['count'], ascending=False).reset_index().drop('index', 1)
frequencies_fig = px.bar(token_counts, x='tokens', y='count')
st.plotly_chart(frequencies_fig, use_container_width=True)


# Think about the metric to measure the demand