import pandas as pd
import datetime
import config
from serpapi import GoogleSearch
from google.cloud import bigquery

# Project Details
project = "trans-gate-374512"
dataset_id = "Singapore_Jobs"
dataset_ref = bigquery.DatasetReference(project, dataset_id)

# Current Date
current_date = datetime.date.today()

def data_analyst_ingestion():
    """Ingest data analyst job data"""

    google_search_api_key = config.config_vars['google_search_api_key']
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

    # Execute the search to retrieve the results
    for x in range(3):
        start = x * 10
        params["start"] = start
        search = GoogleSearch(params)
        results = search.get_dict()
        jobs = results["jobs_results"]
        jobs_df = pd.DataFrame(jobs)
        jobs_df = pd.concat([jobs_df, pd.json_normalize(jobs_df["detected_extensions"])], axis=1).drop("detected_extensions", 1)
        if 'salary' in jobs_df.columns:
            jobs_df = jobs_df.drop('salary', 1)
        
        if x == 0:
            current_day_jobs_df = jobs_df
        else:
            current_day_jobs_df = current_day_jobs_df.append(jobs_df, ignore_index=True)

    # Add date column to dataframe and drop related links and extensions
    current_day_jobs_df['date'] = pd.to_datetime(current_date)
    current_day_jobs_df = current_day_jobs_df.drop('related_links', 1).drop('extensions', 1)

    # Insert data into BigQuery
    client = bigquery.Client()
    table_ref = dataset_ref.table("raw_jobs")
    table = client.get_table(table_ref)
    errors = client.insert_rows_from_dataframe(table, current_day_jobs_df)
    if errors == []:
        print(errors)
        print('Upload failed')
    else: 
        print('Data loaded into table')