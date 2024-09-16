import re
import pandas as pd
import streamlit as st
from scrape import TrendScraper
from rag import RAGTrends
from tweet import generate_tweet
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
import torch
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

st.set_page_config(
    layout="centered",
    page_title="News Digest",
    page_icon="📰",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "a robust tool for trend monitoring and information distillation.",
    },
)

st.title("News Digest")
st.subheader(":red[All in one place.]")

@st.cache_resource
def get_models(embedding_type:str="local"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_api_key = os.getenv("HF_API_KEY")
    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    # Initialize embeddings based on type
    hf_model_name= "sentence-transformers/all-MiniLM-l6-v2"
    if embedding_type == 'local':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings_model = HuggingFaceEmbeddings(model_name=hf_model_name, model_kwargs={'device': device})
    elif embedding_type == 'api':
        embeddings_model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api_key, model_name=hf_model_name)
    else:
        raise ValueError("embedding_type must be either 'local' or 'api'")
    return llm, embeddings_model

llm, embeddings_model = get_models()

@st.cache_resource
def get_scrapers():
    return TrendScraper("realtime"), TrendScraper("daily")

trend_realtime_scraper, trend_daily_scraper = get_scrapers()
def fetch_trend_data():
    gg_daily_data_dict = trend_daily_scraper.fetch_and_parse_xml()
    gg_realtime_data_dict = trend_realtime_scraper.fetch_and_parse_xml()
    return gg_daily_data_dict, gg_realtime_data_dict
@st.cache_data
def get_trend_titles(data_dict):
    return [item['title'] for item in data_dict['rss']['channel']['item']]

# Initialize session state
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = datetime.now()
if 'gg_daily_data_dict' not in st.session_state or 'gg_realtime_data_dict' not in st.session_state:
    st.session_state.gg_daily_data_dict, st.session_state.gg_realtime_data_dict = fetch_trend_data()

if (st.button("Refresh Trends")):
    st.session_state.gg_daily_data_dict, st.session_state.gg_realtime_data_dict = fetch_trend_data()
    st.session_state.last_fetch_time = datetime.now()
    st.session_state.pop('daily_rag_results', None)
    st.session_state.pop('realtime_rag_results', None)
    st.success("Trends refreshed!")

daily_trend_title_list = get_trend_titles(st.session_state.gg_daily_data_dict)
realtime_trend_title_list = get_trend_titles(st.session_state.gg_realtime_data_dict)

st.divider()
col1, col2 = st.columns(2)
with col1:
    top_k_daily = st.slider("Top k daily trends", min_value=1, max_value=len(daily_trend_title_list), value=7)
with col2:
    top_k_realtime = st.slider("Top k realtime trends", min_value=1, max_value=len(realtime_trend_title_list), value=7)

@st.cache_data
def scrape_trends(trends_title_list, _scraper):
    return _scraper.run()
@st.cache_data
def run_rag(_rag_chain_instance, _gg_df, _ddg_df, kws):
    return asyncio.run(_rag_chain_instance.run_rag(_gg_df, _ddg_df))

def extract_domain(url):
    """
    Extracts the domain from a given URL.

    Parameters:
        url (str): The URL from which to extract the domain.

    Returns:
        str or None: The extracted domain if found, otherwise None.
    """
    pattern = re.compile(r"https?://(?:www\.)?([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})")
    match = pattern.search(url)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None
def display_trend_results(rag_results, google_df, ddg_df, trend_type):
    if trend_type == "daily":
        k = top_k_daily
    elif trend_type == "realtime":
        k = top_k_realtime
    else:
        raise ValueError("trend_type must be either 'daily' or'realtime'")
    for i in range(min(k,len(rag_results["Trend_kws"]))):
        with st.expander(f"**{rag_results['Title'][i]}**", expanded=False):
            st.write(rag_results["Summary"][i])
            st.divider()
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                st.write(f"🚀 Traffic: {google_df['traffic'].iloc[i]} Google Searches")
            with col2:
                st.write(f"📅 Date: {pd.Timestamp(google_df['pubDate'].iloc[i]).strftime('%a %d %b %Y, %I:%M%p')}")
            
            domains_list = [extract_domain(link) for link in google_df["url"].iloc[i]]
            st.markdown(f"🌐 References: " + ", ".join([f"[{domains_list[j]}]({google_df['url'].iloc[i][j]})" for j in range(len(google_df['url'].iloc[i][:3]))]))

            col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
            with col1:
                sent_menu = st.selectbox(f"Sentiment", ["positive", "neutral", "negative"], key=f"sentiment_{trend_type}_{i}")
            with col2:
                tone_menu = st.selectbox(f"Tone", ["Professional", "Friendly", "Sarcastic", "Aggressive", "Skeptical", "Shakespearean"], key=f"tone_{trend_type}_{i}")
            with col3:
                gen_twt = st.button(f"Generate Tweet", key=f"tweet_btn_{trend_type}_{i}")
                
            if gen_twt:
                resp = generate_tweet(llm, rag_results["Title"][i], rag_results["Summary"][i], sent_menu, tone_menu)
                st.write(f"#️⃣Tweet:\n {resp.content}")

tab1, tab2 = st.tabs(["Daily Trends", "Realtime Trends"])
with tab1:
    if 'daily_rag_results' not in st.session_state:
        google_daily_df, ddg_daily_df = scrape_trends(daily_trend_title_list, trend_daily_scraper)
        rag_chain = RAGTrends(llm_model=llm, hf_embeddings_model=embeddings_model)
        st.session_state.daily_rag_results = run_rag(rag_chain, google_daily_df, ddg_daily_df, google_daily_df.trend_kws.to_list())
        st.session_state.google_daily_df = google_daily_df
        st.session_state.ddg_daily_df = ddg_daily_df
        st.toast('Daily news digest ready!', icon='🔥')
    
    with st.expander("Show scraping results", expanded=False):
        st.write("Google search daily trends results")
        st.write(st.session_state.google_daily_df.iloc[:top_k_daily,:])
        st.write("")
        st.write("Duck-Duck-Go News results for Google daily trends")
        st.write(st.session_state.ddg_daily_df.iloc[:top_k_daily*3,:])
    
    
    display_trend_results(st.session_state.daily_rag_results, st.session_state.google_daily_df, st.session_state.ddg_daily_df, "daily")

with tab2:
    if 'realtime_rag_results' not in st.session_state:
        google_realtime_df, ddg_realtime_df = scrape_trends(realtime_trend_title_list, trend_realtime_scraper)
        rag_chain = RAGTrends(llm_model=llm, hf_embeddings_model=embeddings_model)
        st.session_state.realtime_rag_results = run_rag(rag_chain, google_realtime_df, ddg_realtime_df, google_realtime_df.trend_kws.to_list())
        st.session_state.google_realtime_df = google_realtime_df
        st.session_state.ddg_realtime_df = ddg_realtime_df
        st.toast('Realtime news digest ready!', icon='🔥')

    
    with st.expander("Show scraping results", expanded=False):
        st.write("Google search realtime trends results")
        st.write(st.session_state.google_realtime_df.iloc[:top_k_realtime,:])
        st.write("")
        st.write("Duck-Duck-Go News results for Google realtime trends")
        st.write(st.session_state.ddg_realtime_df.iloc[:top_k_realtime*3,:])
    
    display_trend_results(st.session_state.realtime_rag_results, st.session_state.google_realtime_df, st.session_state.ddg_realtime_df, "realtime")

st.sidebar.write(f"Last data refresh: {st.session_state.last_fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")