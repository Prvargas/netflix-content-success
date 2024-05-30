#The purpose of this file is to house all of the helper functions used to clean the Netflix engagement data

##FUNCTION 1: a function that will remove all special characters from a string and return it as all lowercase.
import re

def remove_special_characters(string):
    # Define the regular expression pattern for numbers within square brackets
    pattern = r'\[\d+\]'
    
    # Substitute all numbers within square brackets with an empty string
    cleaned_string = re.sub(pattern, "", string)
    
    # Remove any remaining non-alphanumeric characters except spaces
    cleaned_string = re.sub(r'[^a-zA-Z ]', '', cleaned_string)
    
    return cleaned_string.lower()



##FUNCTION 2: a function that will remove the word “season” from the title string.
import re

def remove_season(input_string):
    # Define the regular expression pattern to match "Season" and everything after it,
    # but only if "Season" is not at the very beginning of the string.
    # This version uses a negative lookbehind assertion to ensure "Season" is not directly at the start.
    pattern = r'(?<!^)Season.*'

    # Replace the matched pattern with an empty string
    cleaned_string = re.sub(pattern, '', input_string, flags=re.IGNORECASE)

    return cleaned_string.strip()



##FUNCTION 3: a function to parse the season number from the title string and return on the number as integer.
import re
import numpy as np

def parse_season_number(input_string):
    # Define the regular expression pattern to find "Season" followed by a number
    pattern = r'Season\s+(\d+)'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string, flags=re.IGNORECASE)

    if match:
        # If a match is found, convert the matched group (number) to an integer
        return int(match.group(1))
    else:
        # If no match is found, return NaN
        return np.nan




##FUNCTION 4: This function checks if a title string contains any numerical values and returns a boolean value.
def contains_numbers(string):
  """
  This function checks if a string contains any numerical values.

  Args:
      string: The string to be checked.

  Returns:
      True if the string contains numbers, False otherwise.
  """
  return any(char.isdigit() for char in string)




##FUNCTION 5:  a function that counts the number of characters in the title string and returns an integer.
def char_len(string):
   #Convert to string data type
   string = str(string)

   #Get char count using len function
   string_len = len(string)
   
   return string_len



##FUNCTION 6: Plots a bar chart showing the count of titles that appear exactly once and more than once
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_clean_title_counts(df, column_name):
    """
    Plots a bar chart showing the count of titles that appear exactly once and more than once,
    with percentage labels on the bars.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    column_name (str): The name of the column to analyze.
    """
    # Step 1: Get the value counts of the specified column
    title_counts = df[column_name].value_counts()

    # Step 2: Calculate the counts for titles that appear exactly once and more than once
    count_equal_to_one = (title_counts == 1).sum()
    count_greater_than_one = (title_counts > 1).sum()

    # Create a dataframe for plotting
    count_data = pd.DataFrame({
        'Category': ['Exactly Once', 'More Than Once'],
        'Count': [count_equal_to_one, count_greater_than_one]
    })

    # Calculate the percentages
    total = count_data['Count'].sum()
    count_data['Percentage'] = (count_data['Count'] / total) * 100

    # Step 3: Plot the bar chart
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x='Category', y='Count', data=count_data)

    # Step 4: Add percentage labels on the bars
    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.annotate(f'{height:.0f}\n({height/total:.1%})',
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom')

    # Customize the plot
    plt.title('Count of Clean Titles')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.ylim(0, max(count_data['Count']) * 1.1)  # Add some space above the highest bar

    # Display the plot
    plt.show()

# Example usage:
# Assuming netflix_clean_df is already defined and contains the column 'Clean_Title'
#plot_clean_title_counts(netflix_clean_df, 'Clean_Title')





##FUNCTION 7: Plots a horizontal bar chart of the top N most repeated titles in the specified column
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_repeated_titles(df, column_name, top_n=20):
    """
    Plots a horizontal bar chart of the top N most repeated titles in the specified column,
    excluding blank string values.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    column_name (str): The name of the column to analyze.
    top_n (int): The number of top repeated titles to plot. Default is 20.
    """
    # Step 1: Filter out blank string values
    filtered_df = df[df[column_name] != ""]
    
    # Step 2: Get the value counts of the specified column
    title_counts = filtered_df[column_name].value_counts().head(top_n)
    
    # Create a dataframe for plotting
    count_data = pd.DataFrame({
        'Title': title_counts.index,
        'Count': title_counts.values
    })
    
    # Step 3: Plot the horizontal bar chart
    plt.figure(figsize=(12, 8))
    bar_plot = sns.barplot(x='Count', y='Title', data=count_data, palette='viridis')

    # Step 4: Add count labels on the bars
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_width():.0f}',
                          (p.get_width(), p.get_y() + p.get_height() / 2),
                          ha='left', va='center', xytext=(5, 0), textcoords='offset points')
    
    # Customize the plot
    plt.title(f'Top {top_n} Most Repeated Titles')
    plt.xlabel('Count')
    plt.ylabel('Title')
    
    # Display the plot
    plt.show()


# Example usage:
# Assuming netflix_clean_df is already defined and contains the column 'Clean_Title'
#plot_top_repeated_titles(netflix_clean_df, 'Clean_Title')

