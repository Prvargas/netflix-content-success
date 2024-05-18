
# Enhancing Netflix Engagement Data: Leveraging AI for Content Classification and Analysis


# BACKGROUND

## DATA SOURCE

On December 12th, 2023, Netflix published “What We Watched: A Netflix Engagement Report,” which contained engagement data from January 2023 to June 2023. This comprehensive report, available [here](https://about.netflix.com/en/news/what-we-watched-a-netflix-engagement-report), includes:

- **Title**: The title of the content.
- **Available Globally?**: Whether a title was available globally.
- **Release Date**: The premiere date for any Netflix TV series or film.
- **Hours Viewed**: Hours viewed for every title watched for over 50,000 hours.

In total, this report covers more than 18,000 titles — representing 99% of all viewing on Netflix — and nearly 100 billion hours viewed. The data was downloaded as an Excel file.

![Raw Excel Engagement Data](imgs/01-Raw_Excel_Engagment_Data.png)

### Can the dataset be enhanced for better insights and accuracy?
While the provided data is extensive, it can be enhanced with additional columns such as "is_original" and "content_type." These additional attributes cannot easily be found online for all 18,000 titles. By identifying which titles are Netflix Originals and categorizing the content type (e.g., film, series, documentary), the dataset can be made more usable and insightful for further analysis and strategic decision-making.


## GOAL

The goal of this project is to enhance Netflix engagement data by creating two additional columns: **"is_original"** and **"content_type"**. This will be achieved by leveraging the LangChain Framework, OpenAI ChatGPT API, and Few-Shot Prompting. 

Specifically, the objectives are:

- **"is_original"** - **Identify Netflix Original Content**: Determine if each of the 18,000 titles is an original Netflix production.
- **"content_type"** - **Classify Content Type**: Categorize each title as a film, series, or documentary.

By enriching the dataset with these columns, we aim to provide a more detailed and usable dataset that can offer deeper insights into Netflix's content and viewership patterns.



## VALUE

Enhancing Netflix’s engagement data with additional columns "is_original" and "content_type" offers significant benefits:

- **Improved Data Usability**: The enriched dataset provides more comprehensive insights, allowing for better analysis of content trends and viewership patterns.
- **Strategic Decision-Making**: With clear identification of Netflix Originals and content types, stakeholders can make more informed decisions regarding content acquisition, production, and distribution strategies.
- **Cost-Effective Solution**: Utilizing the LangChain Framework, OpenAI ChatGPT API, and Few-Shot Prompting enables efficient data enhancement at a low cost, saving both time and resources compared to manual data classification.
- **Enhanced Viewer Insights**: Understanding the distribution and performance of original versus licensed content helps in tailoring marketing strategies and improving viewer engagement.
- **Scalability**: The automated approach to data enhancement can be scaled to process larger datasets in the future, ensuring continuous improvement of data quality and usability.





# DATA PREPERATION
## EDA: Raw Engagement Data

### Part 1: Ydata Profiling (Pandas-Profiling)

1. **Loading Data**: The raw engagement data was loaded into a pandas DataFrame using Python and Jupyter Notebook.
2. **Data Types Check**: The DataFrame data types were checked using the `.info()` method.
3. **Data Type Conversion**: The "Release Date" column was converted from object to datetime data type.
4. **Data Profiling**: A low-code data profiling library named "Ydata Profiling" was used to generate a profile report.

**Profile Report Notes**:
- The "Release Date" column contains missing values for a significant portion of the dataset. Only 4,855 out of 18,214 entries have non-null release dates.
- "Hours Viewed" represents the aggregate viewership, reflecting the popularity and engagement of each title.
- The dataset provides insights into global availability, indicating Netflix's distribution strategy.
- This dataset allows for analysis of content trends over time, especially with the inclusion of "Release Date" and "Hours Viewed".

### Part 2: Manual EDA - High Level

Visual insights from the Exploratory Data Analysis (EDA) of the Netflix Engagement Report for 2023:

- **Distribution of Global Availability**: A bar chart reveals a significant difference in the number of titles available globally versus those not available globally. With 4,514 (25%) titles available globally, a larger proportion, 13,700 (75%) titles, are not available for streaming worldwide.
- **Viewership Distribution**: A histogram, using a logarithmic scale due to the wide range of viewership hours, showcases the distribution of "Hours Viewed" across titles. Most titles have relatively low viewership hours, with a few exceptions reaching extremely high viewership, indicating a long-tail distribution where a small number of titles capture the majority of viewing hours.
- **Distribution of Release Years for Netflix Titles**: A bar chart shows annual title releases from 2010 to 2023, revealing a steady increase that peaks in 2022. Data for 2023 only covers January to June.

### Part 3: Viewership Distribution: Logarithmic Scale

**What is a logarithmic scale?**

A logarithmic scale is a way of displaying numerical data over a very wide range of values in a compact manner. In such a scale, each unit increase on the axis corresponds to a multiplication of the value it represents, rather than a simple addition.

**Takeaway from observing "Hours Viewed" using a logarithmic scale**:

- **Identifying Long-tail Distribution**: Highlights the long-tail distribution of viewership, where a small number of titles accumulate a disproportionately large number of viewing hours, while the vast majority have significantly fewer hours.
- **Enhanced Visibility for Low-Viewership Titles**: Makes it easier to see the distribution of titles with relatively low viewership hours.
- **Improved Interpretability**: Aids in interpreting data with a wide range, making it possible to observe and analyze both the high-performing and low-performing titles within the same visual space.



## Data Cleaning and Preprocessing

### Overview

The data cleaning and preprocessing steps were critical in preparing the raw engagement data for enhancement. Custom functions were created to clean and preprocess the title strings, ensuring consistency and removing unnecessary information.

### Steps

1. **Title String Cleaning**: 
    - Created custom functions to clean the strings in the "Title" column and create a new column "Clean_Title".
    - Removed season info from titles and created another column "Season_Num" to record the season number.
    - Removed all special characters that were not alphanumeric and converted all letters to lowercase.
    - Filtered out all "Clean_Title" entries with a character length of less than 10, except for those with "Hours Viewed" greater than or equal to 700,000 (the median for the entire dataset). A total of 465 records met the exception criteria.

### Helper Functions

- **remove_special_characters**: Removes all special characters from a string and converts it to lowercase.
- **remove_season**: Removes the word "season" from the title string.
- **parse_season_number**: Parses the season number from the title string and returns it as an integer.
- **contains_numbers**: Checks if a title string contains any numerical values and returns a boolean value.
- **char_len**: Counts the number of characters in the title string and returns an integer.

### Data Cleaning Steps

1. **remove_special_characters**: Applied to the "Title" column to create the "Clean_Title" column.
2. **remove_season**: Applied to remove the word "season" from titles.
3. **parse_season_number**: Extracted season numbers from titles and recorded them in the "Season_Num" column.
4. **contains_numbers**: Checked for numerical values in titles to aid in cleaning.
5. **char_len**: Used to filter out short titles, ensuring only meaningful titles were retained.

### Final Cleaned Data

- Grouped by the "Clean_Title" column to avoid duplicate classification and save on API costs.
- Ensured the cleaned and preprocessed data was ready for the enhancement process using the LangChain Framework and OpenAI API.

By following these steps, the raw data was transformed into a clean and structured format, enabling accurate and efficient data enhancement.




# DATA ENHANCEMENT
## Data Enhancement using LangChain Framework, OpenAI API, & Few Shot Prompting

### Overview

The data enhancement process involved using the LangChain Framework, OpenAI ChatGPT API, and Few-Shot Prompting to classify each title as either Netflix Original Content or not, and to determine the content type (film, series, or documentary).

### Steps

1. **Few-Shot Prompt Creation**:
    - Created a few-shot prompt with examples using the LangChain class `FewShotPromptTemplate`.
    - Provided examples that asked questions about a title to guide the AI in making classifications.

2. **Instantiate ChatOpenAI Object**:
    - Instantiated a `ChatOpenAI` object from LangChain to be used as the LLM Model with the `api_key` and model name.

3. **Load Output Parser**:
    - Loaded a parser object `StrOutputParser()` from the LangChain library to parse the responses from the OpenAI API.

4. **Create Few-Shot LLMChain**:
    - Created a LangChain few-shot `LLMChain` using `FewShotPromptTemplate`, `ChatOpenAI`, and `StrOutputParser()` as the components.

5. **Enhance Data**:
    - Used the few-shot `LLMChain` to enhance the Netflix engagement data by returning if the title is "is_original" and determining the "content_type".
    - Parsed the AI responses to extract the desired information and updated the dataset accordingly.

### Example AI Response

- **Original Netflix content**: No
- **Are follow-up questions needed here**: Yes
- **Follow-up**: Is title 'the boss baby' an ongoing series, a film, or a documentary?
- **Intermediate answer**: Film
- **Final answer**: `{"title":"the boss baby", "is_original":"No", "content_type":"Film"}`

### Example Parsed Output

- `{"title":"the boss baby", "is_original":"No", "content_type":"Film"}`

### Process Details

1. **Few-Shot Prompting**:
    - Created detailed few-shot prompts to provide context and guide the AI in classifying titles accurately.
    - Example prompts included questions about the title's originality and content type.

2. **Model Setup**:
    - Configured the `ChatOpenAI` object with appropriate API keys and model specifications to interact with the OpenAI ChatGPT API.
    
3. **Data Parsing**:
    - Utilized `StrOutputParser()` to parse the output from the AI, ensuring the results were in a structured format for easy integration into the dataset.
    
4. **Data Update**:
    - Enhanced the original dataset by adding two new columns: "is_original" and "content_type", based on the AI's responses.

By following these steps, the engagement data was successfully enhanced, providing more detailed and valuable insights into Netflix's content library.



# Conclusion

This project successfully enhanced Netflix's engagement data by leveraging the LangChain Framework, OpenAI ChatGPT API, and Few-Shot Prompting. The enhancement process involved creating two additional columns, "is_original" and "content_type," for each of the 18,000 titles in the dataset. 

### Key Achievements

- **Data Usability**: The enriched dataset provides comprehensive insights into the nature of Netflix's content, facilitating more in-depth analysis and strategic decision-making.
- **Efficient Classification**: Automated the classification of Netflix Originals and content types, reducing the time and resources required for manual classification.
- **Scalable Solution**: Established a scalable approach that can be applied to larger datasets in the future, ensuring continuous improvement of data quality.
- **Informed Decision-Making**: The enhanced data supports Netflix's distribution strategy and content acquisition decisions by providing clear distinctions between original and licensed content and detailed content types.
- **Cost-Effective**: Utilized advanced AI techniques to enhance data at a low cost, demonstrating the value of integrating AI-driven solutions into data analysis workflows.

### Final Outcome

The project resulted in a more valuable and usable dataset, with the following new columns:

- **is_original**: Indicates whether a title is original Netflix content.
- **content_type**: Categorizes the title as a "Film", "Series", or "Documentary".

These enhancements allow for better analysis of content trends, viewership patterns, and strategic planning, ultimately contributing to more informed business decisions for Netflix.




## NEXT STEPS

The next steps of this project involve incorporating additional features to further enhance the dataset and enable predictive analytics to forecast the success of Netflix original content based on "Hours Viewed". These additional features include:

- **Genre**: Classify each title into one or more genres to analyze content trends and viewer preferences.
- **IMDb Data**: Integrate IMDb ratings and reviews to provide a comprehensive view of the content's reception and popularity.
- **Rating**: Include content ratings (e.g., PG, R) to understand the target audience and content suitability.
- **Keywords**: Extract and analyze keywords from titles and descriptions to identify common themes and topics.
- **Description**: Incorporate content descriptions to provide context and detailed information about each title.

### Goals for Next Steps

- **Feature Enrichment**: Enhance the dataset with additional attributes to provide a richer and more detailed dataset.
- **Predictive Analytics**: Develop models to predict the success of Netflix original content based on various features, including "Hours Viewed".
- **Trend Analysis**: Analyze trends and patterns across different genres, ratings, and keywords to gain deeper insights into viewer preferences.
- **Viewer Engagement**: Understand the factors that drive viewer engagement and content popularity to inform content acquisition and production strategies.

By incorporating these additional features, we aim to create a comprehensive dataset that supports advanced analytics and predictive modeling, ultimately helping Netflix make more informed decisions about content strategy and viewer engagement.


