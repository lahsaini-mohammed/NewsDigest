from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define output schema for llm
class SummmaryWithTitle(BaseModel):
    '''Article summary and title.'''
    title: str
    summary: str

class RAGTrends:
    def __init__(self, llm_model, hf_embeddings_model):
        self.llm = llm_model
        self.embeddings_model = hf_embeddings_model
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful assistant that specializes in article summarization.
            Your task is to summarize a given text article without referring to it, and generate a title for it.
            The summary should be between 4 and 10 lines, depending on how many details are given.
            If the provided article doesn't contain coherent and meaningful content, just return an empty response.
            """),
            ("human", "Article: {article}"),
        ])
        self.dict_schema = convert_to_openai_tool(SummmaryWithTitle)
        self.faiss_cache: Dict[str, FAISS] = {}
    @staticmethod
    async def _fetch(session: aiohttp.ClientSession, url: str, timeout: int = 3) -> Optional[str]:
        try:
            async with session.get(url, timeout=timeout) as response:
                return await response.text()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout occurred while fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    async def _fetch_all(urls: List[str], max_concurrent: int = 100, timeout: int = 3) -> List[Optional[str]]:
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [RAGTrends._fetch(session, url, timeout) for url in urls]
            return await asyncio.gather(*tasks)

    @staticmethod
    def _parse_content(html: Optional[str], url: str) -> Optional[Document]:
        if html is None:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        metadata = {
            "source": url,
            "title": soup.title.string if soup.title else "No title",
        }
        return Document(page_content=text, metadata=metadata)

    @staticmethod
    async def url_loader(url_list: List[str], max_concurrent: int = 100, timeout: int = 3) -> List[Document]:
        html_contents = await RAGTrends._fetch_all(url_list, max_concurrent, timeout)
        
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4)*5) as executor:
            documents = list(executor.map(RAGTrends._parse_content, html_contents, url_list))
        
        return [doc for doc in documents if doc is not None]
    @staticmethod
    def _rec_splitter(url_doc_list: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            add_start_index=True
        )
        return text_splitter.split_documents(url_doc_list)
    @staticmethod
    def _retrieved_docs_parser(ret_doc_list: List[Document]) -> str:
        """
        Parses a list of retrieved documents and extracts the meaningful sentences from the article.

        Args:
            ret_doc_list (list): A list of retrieved documents.

        Returns:
            str: The meaningful sentences from the article, joined by newline characters.

        Description:
            This function takes a list of retrieved documents and extracts the meaningful sentences from the scraped article text (not cleaned).
            It joins the page content of each document into a single string, replacing consecutive newline characters with a dot followed by a space.
            Then, it splits the article into sentences using a regular expression pattern.
            The function filters out sentences that have fewer than 5 words and joins the remaining sentences into a single string, separated by newline characters.
            The resulting string contains the meaningful sentences from the article.
        """
        ret_article = "\n".join([doc.page_content for doc in ret_doc_list])
        ret_article = re.sub(r'\n+', '. ', ret_article)
        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][A-Z]\.)(?<=\.|\?|!|\n)\s*')
        ret_article_sentences = sentence_pattern.split(ret_article)
        ret_article_meaningful_sentences = [s for s in ret_article_sentences if len(s.split()) > 5]
        return '\n'.join(ret_article_meaningful_sentences)
    
    async def _run_rag_chain_once(self, trend: str, google_df: Any, ddg_df: Any, retrieved_docs_parser_runnable: Any, structured_output_llm: Any) -> Dict[str, str]:
        """
        Runs a single iteration of the RAG pipeline for a given trend.

        Args:
            trend (str): The trend keyword.
            google_df (pandas.DataFrame): The Google Trends data.
            ddg_df (pandas.DataFrame): The DuckDuckGo news data.
            reteived_docs_parser_runnable (callable): The function to parse retrieved documents.
            structured_output_llm (callable): The function to generate structured output using an LLM.

        Returns:
            Any: The results of the RAG pipeline.
        """
        logger.info(f"Starting RAG pipeline for keyword: {trend}")
        start = time.time()

        logger.info("Scraping Articles")
        df_trend = google_df[google_df["trend_kws"] == trend]
        url_list = df_trend['url'].iloc[0]
        
        url_docs = await self.url_loader(url_list)
        
        for doc in url_docs:
            if (not doc.page_content) and (doc.metadata["source"] in ddg_df.url.to_list()):
                article_body_index = ddg_df['url'].to_list().index(doc.metadata["source"])
                doc.page_content += ddg_df['body'][article_body_index]
        
        scraping_checkpoint = time.time()
        scraping_dur = scraping_checkpoint - start

        logger.info("Creating or retrieving FAISS vector store")
        cache_key = frozenset(url_list)
        if cache_key not in self.faiss_cache:
            splits_docs = self._rec_splitter(url_docs)
            self.faiss_cache[cache_key] = FAISS.from_documents(splits_docs, self.embeddings_model)

        faiss_db = self.faiss_cache[cache_key]
        faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={'k': 5})
        ret_query = '\n'.join(df_trend['title'].iloc[0])
        
        faiss_checkpoint = time.time()
        faiss_dur = faiss_checkpoint - scraping_checkpoint

        logger.info("Performing RAG")
        rag_chain = (
            faiss_retriever
            | {"article": retrieved_docs_parser_runnable}
            | self.prompt_template
            | structured_output_llm
        )
        rag_results = await rag_chain.ainvoke(ret_query)
        
        end = time.time()
        chain_dur = end - faiss_checkpoint
        logger.info(f"Scrape: {scraping_dur:.2f}s, Faiss: {faiss_dur:.2f}s, Chain: {chain_dur:.2f}s")
        
        return rag_results
    
    async def run_rag(self, google_df: Any, ddg_df: Any) -> Dict[str, List[str]]:
        """
        Runs the RAG (Retrieval-based Article Generation) process for each trend keyword in the given Google DataFrame.

        Args:
            google_df (DataFrame): The Google DataFrame containing trend keywords and their corresponding URLs.
            ddg_df (DataFrame): The DataFrame containing DuckDuckGo search results.

        Returns:
            dict: A dictionary containing the trend keywords, titles, and summaries generated by the RAG process.
                - "Trend_kws" (list): A list of trend keywords.
                - "Title" (list): A list of generated titles.
                - "Summary" (list): A list of generated summaries.
        """
        trend_kws = google_df.trend_kws.to_list()
        retrieved_docs_parser_runnable = RunnableLambda(self._retrieved_docs_parser)
        structured_output_llm = self.llm.with_structured_output(self.dict_schema)

        async def process_trend(trend_kw):
            if google_df[google_df["trend_kws"] == trend_kw]['url'].iloc[0]:
                return await self._run_rag_chain_once(trend_kw, google_df, ddg_df, retrieved_docs_parser_runnable, structured_output_llm)
            else:
                return {'title': trend_kw, 'summary': 'Not enough information yet!'}

        results = await asyncio.gather(*[process_trend(trend_kw) for trend_kw in trend_kws])

        return {
            "Trend_kws": trend_kws,
            "Title": [result['title'] for result in results],
            "Summary": [result['summary'] for result in results]
        }