import re
import requests
import pandas as pd
import xmltodict
from duckduckgo_search import DDGS


class TrendScraper():
    def __init__(self, trend_type:str="realtime"):
        self.domains_skip_list = ["msn.com", "nytimes.com", "washingtonpost.com", "e360.yale.edu", "star-telegram.com", "charlotteobserver.com"]
        if trend_type == "daily":
            self.rss_xml_url = 'https://trends.google.com/trends/trendingsearches/daily/rss?geo=US'
        elif trend_type == "realtime":
            self.rss_xml_url = 'https://trends.google.com/trending/rss?geo=US'
        else:
            raise ValueError("trend_type must be either 'daily' or 'realtime'.")
        
    def fetch_and_parse_xml(self):
        """
        Fetches XML data (Get request) from a given URL and parses it into a dictionary (using xmltodict).
        Args:
            rss_xml_url (str): The URL of the RSS XML feed to fetch and parse. Defaults to realtime rss feed.
            region (str): The region code to append to the URL. Defaults to 'US'.

        Returns:
            dict: A dictionary representation of the parsed XML data.
                Returns None if an error occurs.

        Raises:
            requests.exceptions.RequestException: If an error occurs while fetching the XML data.
            xmltodict.expat.ExpatError: If an error occurs while parsing the XML data.
            Exception: if any other unexpected error occurs.
        """
        try:
            # Fetch the XML data
            response = requests.get(self.rss_xml_url)
            response.raise_for_status()  # Raise an error for bad status codes
            xml_data = response.content
            # Parse the XML data and convert it to a dictionary
            data_dict = xmltodict.parse(xml_data)
            return data_dict
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the XML data: {e}")
        except xmltodict.expat.ExpatError as e:
            print(f"Error parsing the XML data: {e}")
        except Exception as e:
            print(f'Error: {e}')

    def clean_text(self, text):
        """
        Clean and normalize the input text by removing special characters and non-ASCII characters.#+

        This function performs the following operations:
        1. Replaces HTML single quote code with an actual single quote.
        2. Removes non-English (non-ASCII) characters.
        3. Removes special characters and punctuation, keeping only letters, numbers, and spaces.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned and normalized text.
        """
        text = text.replace("&#39;", "'")
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def create_google_dataframes(self, trends_dict):
        """
        Creates a pandas DataFrame from a dictionary containing Google Trends data.

        Args:
            trends_dict (dict): A dictionary containing the Google Trends data. It should have the following structure:
                - 'rss' (dict): A dictionary containing the RSS feed data.
                    - 'channel' (dict): A dictionary containing the channel data.
                        - 'item' (list): A list of dictionaries containing the individual trend data.
                            - 'title' (str): The title of the trend.
                            - 'ht:approx_traffic' (int): The approximate traffic for the trend.
                            - 'pubDate' (str): The publication date of the trend.
                            - 'ht:news_item' (list or dict): A list or dictionary containing the news items for the trend.

            Note: The 'ht:news_item' field can be either a list or a dictionary. If it is a list, each item in the list is a dictionary with the following keys:
                - 'ht:news_item_url' (str): The URL of the news item.
                - 'ht:news_item_title' (str): The title of the news item.

                If it is a dictionary, it has the following keys:
                - 'ht:news_item_url' (str): The URL of the news item.
                - 'ht:news_item_title' (str): The title of the news item.

        Returns:
            google_trends_df (pandas.DataFrame): A DataFrame containing the Google Trends data. It has the following columns:
                - 'trend_kws' (list): The titles of the trends.
                - 'traffic' (list): The approximate traffic for the trends.
                - 'pubDate' (list): The publication dates of the trends.
                - 'url' (list): The URLs of the news items for each trend.
                - 'title' (list): The titles of the news items for each trend.
                - Note: The 'url' and 'title' columns may contain multiple values for each trend if there are multiple news items.

        """
        google_trends_dict = {"trend_kws":[], "traffic":[], "pubDate":[], "url":[], "title":[]}
        for trend in trends_dict['rss']['channel']['item']:
            try:
                google_trends_dict['trend_kws'].append(trend['title'])
                google_trends_dict['traffic'].append(trend['ht:approx_traffic'])
                google_trends_dict['pubDate'].append(trend['pubDate'])
                if isinstance(trend['ht:news_item'], list):
                    google_trends_dict['url'].append([news_item['ht:news_item_url'] for news_item in trend['ht:news_item']])
                    google_trends_dict['title'].append([news_item['ht:news_item_title'] for news_item in trend['ht:news_item']])
                else:
                    google_trends_dict['url'].append([trend['ht:news_item']['ht:news_item_url']])
                    google_trends_dict['title'].append([trend['ht:news_item']['ht:news_item_title']])
            except Exception as e:
                print(f"Error processing trend '{trend}': {e}")
        google_trends_df = pd.DataFrame(google_trends_dict)
        # clean each title
        google_trends_df["title"] = google_trends_df["title"].map(lambda links: [self.clean_text(link) for link in links])

        # remove urls that contain domains from domains_skip_list
        for i in range(google_trends_df.shape[0]):
            for j, url in enumerate(google_trends_df.loc[i,"url"]):
                if any(domain in url for domain in self.domains_skip_list):
                    google_trends_df.loc[i,"url"].pop(j)
                    google_trends_df.loc[i,"title"].pop(j)
        return google_trends_df
    
    def create_ddg_dataframe(self, google_trends_df):
        """
        Creates a DataFrame of DDG news results for each trend keyword in the given Google trends DataFrame.

        Args:
            google_trends_df (pandas.DataFrame): The Google trends DataFrame containing the trend keywords.

        Returns:
            pandas.DataFrame: The DataFrame of DDG news results, with each row representing a news result.
        """

        trends_news = []
        for trend_kw in google_trends_df.trend_kws.to_list():
            # searches for news article with the given keyword, using worldwide region, moderate safe search, and a maximum of 7 results.
            results = DDGS().news(keywords=trend_kw, max_results=7)
            filtered_results = [res for res in results if not any(domain in res['url'] for domain in self.domains_skip_list)]
            filtered_results = list(map(lambda d: {'trend_kws':trend_kw, **d}, filtered_results[:3]))
            trends_news.extend(filtered_results)
        
        trends_ddg_news_df = pd.DataFrame(trends_news)
        return trends_ddg_news_df
    
    def update_google_trends_with_ddg_news(self, google_trends_df, trends_ddg_news_df):
        """
        Update the Google Trends DataFrame with additional URLs and titles from DuckDuckGo news search results.

        This function iterates through each trend in the Google Trends DataFrame and finds corresponding
        news articles from the DuckDuckGo news search results. It then appends these additional URLs and
        titles to the existing lists in the Google Trends DataFrame.

        Args:
            google_trends_df (pandas.DataFrame): DataFrame containing Google Trends data.
                Expected to have columns 'trend_kws', 'url', and 'title'.
            trends_ddg_news_df (pandas.DataFrame): DataFrame containing DuckDuckGo news search results.
                Expected to have columns 'trend_kws', 'url', and 'title'.

        Returns:
            None. The function modifies the google_trends_df in-place.
        """
        for i, trend in enumerate(google_trends_df.trend_kws):
            url_list = trends_ddg_news_df[trends_ddg_news_df["trend_kws"]==trend]['url'].to_list()
            title_list = trends_ddg_news_df[trends_ddg_news_df["trend_kws"]==trend]['title'].to_list()
            google_trends_df.loc[i, "url"].extend(url_list)
            google_trends_df.loc[i, "title"].extend(title_list)
        return google_trends_df
    
    def run(self):
        trends_dict = self.fetch_and_parse_xml()
        google_trends_df = self.create_google_dataframes(trends_dict)
        trends_ddg_news_df = self.create_ddg_dataframe(google_trends_df)
        google_trends_df = self.update_google_trends_with_ddg_news(google_trends_df, trends_ddg_news_df)
        return google_trends_df, trends_ddg_news_df