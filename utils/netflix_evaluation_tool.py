#The purpose of this file is to house all of the helper functions used to evaluate the Netflix engagement data that has been enahnced.



## FUNCTION 1 - Scrape all tables from provided url
import requests
from bs4 import BeautifulSoup

def scrape_tables_from_url(url, element='table'):
    """
    Scrapes all <table> elements from the given URL and stores them in a list.

    Parameters:
    - url (str): The URL to scrape tables from.

    Returns:
    - list: A list of BeautifulSoup objects representing each table found.
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check for successful response
    if response.status_code != 200:
        return f"Failed to retrieve content from {url}, status code: {response.status_code}"

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all <table> elements in the HTML
    tables = soup.find_all(element)

    # Initialize a list to store the tables for later review
    table_list = []

    # Iterate over each table and append it to the table_list
    for table in tables:
        table_list.append(table)

    # Optionally, print the number of tables collected
    print(f'Collected {len(table_list)} tables.')

    return table_list




## FUNCTION 2 - Function to convert HTML table to dataframe
#THIS ONE
#Function to convert HTML table to dataframe

import pandas as pd
from bs4 import BeautifulSoup

def html_table_to_dataframe(html):
    #Convert to string
    html = str(html)

    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', class_='wikitable sortable')

    if table is None:
        raise ValueError("Table not found in HTML content")

    # Attempting to find the first row of the table to extract headers
    first_row = table.find('tr')
    if first_row is None:
        raise ValueError("No rows found in the table")

    headers = [th.text.strip() for th in first_row.find_all('th')]

    data = []
    # Processing each row starting from the second one since we assume the first one contains headers
    for row in table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])  # This will include both td and th elements
        row_data = [cell.text.strip() for cell in cells if cell]
        if len(row_data) != len(headers):
            # Optionally handle the mismatch; for now, we'll just print a warning
            print(f"Warning: row with data {row_data} has a different number of columns than the headers")
        else:
            data.append(row_data)

    df = pd.DataFrame(data, columns=headers)
    return df




## FUNCTION 3 - Pipeline to place entire webscraping and formatting into one helper function
def webscrape_table_pipeline(wikipedia_url):
  #Webscrape URL for tables
  table_lst = scrape_tables_from_url(wikipedia_url)

  #empty list to capture table_dfs
  df_list = []

  #Loop though table list converting to df
  for table in table_lst:
    try:
      table_df = html_table_to_dataframe(table)
      print(table_df.shape, '\n\n')
      df_list.append(table_df)
    except:
      pass


  #output_df = pd.concat(df_list)

  return df_list



## FUNCTION 4 - Formats a list of Netflix DataFrames to ensure they all contain 'Title', 'Genre', and 'Premiere' columns.
import numpy as np
import pandas as pd

def format_netflix_dfs(netflix_orig_df_list):
    """
    Formats a list of Netflix DataFrames to ensure they all contain 'Title', 'Genre', and 'Premiere' columns.

    Parameters:
    - netflix_orig_df_list (list of pd.DataFrame): The list of DataFrames to format.

    Returns:
    - list: A list of formatted DataFrames with the specified columns.
    """
    format_netflix_df_list = []

    # Loop through df list pulling needed titles
    for df in netflix_orig_df_list:
        try:
            format_netflix_df_list.append(df[['Title', 'Genre', 'Premiere']])
        except KeyError as e:
            missing_column = str(e).split("'")[1]  # Extract the missing column name from the error message
            print(f"Adding {missing_column} to dataframe")
            df[missing_column] = np.nan
            format_netflix_df_list.append(df[['Title', 'Genre', 'Premiere']])

    print('TOTAL DATAFRAMES: ',len(format_netflix_df_list))

    #Concat into 1 df
    final_format_df = pd.concat(format_netflix_df_list)

    return final_format_df



#FUNCTION 5 -  Tool to visual model accuracy
import pandas as pd
import matplotlib.pyplot as plt

def plot_gpt_validation_pie_chart(df, column_name):
    """
    Creates and displays a pie chart for the given column in the DataFrame
    which contains only True or False values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to be visualized.
    """
    # Counting the True and False values
    validation_counts = df[column_name].value_counts()

    # Ensure the order of colors corresponds to True and False
    colors = ['#4CAF50' if val else '#FF5722' for val in validation_counts.index]

    # Creating the pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(
        validation_counts, 
        labels=validation_counts.index.map({True: 'Correct', False: 'Incorrect'}), 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors, 
        shadow=True, 
        explode=(0.05, 0.05),  # Slightly separate the slices
        wedgeprops={'edgecolor': 'black'}
    )

    # Adding a legend and a title
    plt.legend(labels=validation_counts.index.map({True: 'Correct', False: 'Incorrect'}), loc='upper right')
    plt.title(column_name, fontsize=14)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the plot
    plt.show()
