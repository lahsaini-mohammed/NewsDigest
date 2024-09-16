# News Digest

## Overview

This project is an application that retrieves trending topics from *Google Trends*, scrapes relevant news articles, generates summaries, and even creates tweets based on user-defined sentiment and tone. It uses scraping tools, a Retrieval-Augmented Generation (RAG) pipeline for summarization, and leverages multiple machine learning models for embedding, summarization, and text generation.

---

## Objectives

1. **Scrape Trending Topics**: Automatically retrieve trending topics from **Google Trends**.
2. **Efficient Article Scraping**: Scrape news articles related to the trending topics using a robust scraping pipeline.
3. **Article Summarization**: Summarize articles and generate concise, informative summaries using **RAG (Retrieval-Augmented Generation)**.
4. **Tweet Generation**: Create tweets related to each trending topic with customizable sentiment and tone.

---

## Installation and Setup

To run this project locally, follow these steps:

#### 1 Clone the repository and navigate to directory

#### 2 create a virtual environment using conda or venv and activate it.

#### 3 install dependencies `pip install -r requirements.txt`

#### 4 if you want to use Ollama make sure it's installed and running (you need to have the used model downloaded `ollama run llama3.1`). if you want to use Groq, huggingface infernece api or other providers make sure to get the needed api keys

#### 5 run web app `streamlit run app.py`
## Key Steps and Workflow

### 1. **Trending Topic Extraction**
   - **Google Trends** is used to get real-time trending topics via **RSS XML** feeds.
   - The extracted data is parsed and cleaned, resulting in a structured **Google Trends DataFrame** containing topic details.
   
### 2. **Web Scraping for Articles**
   - Trending topics are sent as queries to the **DuckDuckGo** (DDG) search engine.
   - A maximum of 7 news articles are retrieved per topic and filtered to exclude unwanted domains (determined by trial-and-error).
   - scrapping using beautifulsoup and aiohhtp .

### 3. **Article Content Processing**
   - Each scraped article undergoes content splitting using a **recursive text splitter**, which chunks the document for further processing.
   - The chunks are indexed in a **FAISS Vector Store** to perform similarity searches based on article titles.

### 4. **Summarization via RAG Pipeline**
   - A **RAG (Retrieval-Augmented Generation)** pipeline is employed for article summarization:
     - Scraped content is retrieved from the vector store.
     - Relevant document chunks are passed to the **Llama 3.1** model to generate a structured summary.
     - A system message is used to guide the model: "You are a helpful assistant that specializes in article summarization..."
   
### 5. **Tweet Generation (Optional)**
   - Users can generate a tweet for each topic, specifying the **sentiment** and **tone** of the tweet (e.g., positive, neutral, negative).
   - A prompt template is used to provide the tweet context (title and summary) and format the output.

---

## Tools and Technologies

1. **Google Trends** – Used to extract trending topics.
2. **DuckDuckGo Search API** – Fetches news articles for each trending topic.
3. **Pytorch, trensformers, sentence-transformers, Langchain** – for document vectorization.
4. **FAISS** – For vector indexing and similarity search of document chunks.
5. **Llama 3.1** – Summarization and text generation model, integrated with **Langchain** for structured output.
6. **Streamlit** – Used for web application UI with caching for faster execution.

---

## Challenges Faced and Solutions

### 1. **Scraping Accuracy vs. Speed**
   - **Challenge**: Articles are scraped from a variety of websites, each with different structures and interfaces, making scraping inconsistent.
   - **Solution**: gracfully handle errors to not stop execution and implement a skip domain list to avoid problematic websites: 
   
   - **Challenge**: the scraping pipeline can be slow, especially when using Selenium, which limits the system’s real-time scalability for high-volume data scraping.
   - **Solution**: rely on asynchronous http requests and asychronous code.

### 2. **Handling Article Noise**
   - **Challenge**: Scraped content includes irrelevant text (menus, ads, footers, etc.), requiring careful cleaning.
   - **Solution**: 
     - Performed **similarity search** on document chunks using the **article title** to filter out unrelated content before summarization.
     - If a document split returns empty or incoherent content, the Llama model returns an empty response, ensuring that invalid documents don't interfere with the summarization process.

### 3. **Scalability**
   - **Challenge**: Scraping multiple articles and generating summaries for numerous trending topics can become resource-intensive.
   - **Solution**: Vectorizing documents and chunking articles reduced the model's input size, improving both memory and processing efficiency.

### 4. **Structured Output for Summaries**
   - **Challenge**: Generating structured summaries using a large language model (LLM).
   - **Solution**: Defined a **custom system prompt** and used **Llama 3.1** (with OpenAI-compatible tool) to ensure structured output.


---

## Critical Analysis

### Strengths:
- **Automation**: End-to-end automation of the process, from topic retrieval to article summarization and tweet generation.
- **Summarization Accuracy**: The RAG pipeline ensures that only relevant parts of the article are processed, improving the quality and speed of the summaries.

### Weaknesses:
- **Scraping Reliability**: the pipeline may still struggle with websites that heavily rely on complex JavaScript or anti-bot measures and websites with newsletter subscription pop-up.
- **Article Quality**: Not all articles scraped are of high quality or relevance, especially if scraping based on search results from DuckDuckGo. There is a reliance on cleaning the article content and ensuring only relevant splits are retrieved.
- **Model Limitations**: While Llama 3.1 is effective, using a more powerful model like GPT-4 could potentially improve the quality of summaries and structured outputs, though it may come with higher cost. Also The Llama model might limit the length of summaries due to token size constraints. This can lead to overly concise or incomplete summaries.
---

## Future Work and Perspectives

1. **Improving Speed and Efficiency**:

    - Log key timing data, such as the time taken for scraping (Time to gather articles from DuckDuckGo and Google Trends.), FAISS indexing (Time taken to retrieve relevant document splits using FAISS), and LLM chain execution (Time spent generating summaries using the Llama 3.1 model). You can use this log data to analyze and optimize the performance of various stages.

   - use pre-scraped datasets (e.g., via news APIs) to improve response time.
   - FAISS indexing is optimized for fast retrieval but can still be tuned by modifying the number of article splits or embeddings.

2. **Advanced Content Filtering**:
   - Implement advanced filtering techniques such as **NLP-based content classifiers** to better remove irrelevant content and focus only on the main article body.

3. **Better Language Models**:
   - Test and integrate newer language models (e.g., **GPT-4**, **Claude**) for better summarization quality, structured output, and contextual understanding.
   - Introduce multilingual support to expand the range of articles that can be processed and summarized.

4. **Enhance Tweet Generation**:
   - Improve tweet generation by fine-tuning sentiment and tone handling, perhaps incorporating a fine-tuned model for the task.
---

This pipeline demonstrates the power of combining **web scraping**, **retrieval-augmented generation**, and **large language models** to automate the extraction and summarization of real-time data from the web. With additional optimizations, it has the potential to be a robust tool for trend monitoring and information distillation.