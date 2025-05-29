#!/usr/bin/env python3
"""
🐦 Twitter Context Generator v2.0
System for generating contextual tweets based on market data
"""

import os
import json
import logging
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load context information
with open("src/info.txt", "r") as file:
    docs = file.read()

# Load market scenarios
with open("src/market_data.json", "r") as file:
    market_data = json.load(file)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

all_splits = text_splitter.split_text(docs)

# Create vector store
semantic_chunk_vectorstore = FAISS.from_texts(all_splits, OpenAIEmbeddings(model="text-embedding-3-small"))

semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the prompt template
rag_template = """
You are **(Name)**, the official X bot for Name. Using the inputs below, craft a single, natural-sounding tweet:

**Inputs:**
- Market Data: `{market_data}`
- User Query: `{query}`
- Context: `{context}`

**Tweet requirements:**
1. ≤ 180 characters
2. Max. 2 short sentences
3. Exactly 1 hashtag: `#(Hashtag)`. Order is not important.
4. No direct CTAs ("join," "buy," etc.); spark curiosity
5. Witty, sarcastic, with cultural references—never robotic or promotional
6. Avoid repetition and forced phrasing
7. Avoid talking about the price of Bitcoin, but discuss the market mood

**Example styles:**
- "Bitcoin's dipping? Time to show who really stinks up the market. #(Hashtag)"
- "The Federal Reserve is raising interest rates? Time to show who really stinks up the market. #(Hashtag)"

**Dynamic tone (based on `market_data`, `mood`, `trending`):**
- Bearish → stoic, reassuring
- Bullish → hype, energetic
- Sideways → humorous, motivational
- + tweaks for `mood`/`trending` as before
- Talk about the facts of news or relevant events on {market_data}

**Deliverable:** Only the final tweet text, without quotes or explanations.
"""

prompt = ChatPromptTemplate.from_template(rag_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def get_context(_):
    return semantic_chunk_retriever.invoke("All about (Name)")

chain = (
    {
        "query": RunnablePassthrough(),
        "context": get_context,
        "market_data": lambda x: x["market_data"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Function to print grid header
def print_grid_header():
    print("\n" + "="*120)
    print(f"{'SCENARIO':<20} | {'MARKET':<30} | {'MOOD':<20} | {'TRENDING':<30} | {'EVENT':<20}")
    print("="*120)

# Function to print grid row
def print_grid_row(scenario):
    trending = ", ".join(scenario["trending_x"])
    print(f"{scenario['name']:<20} | {scenario['crypto_market']:<30} | {scenario['market_mood']:<20} | {trending:<30} | {scenario['global_event']:<20}")
    print("-"*120)

# Main execution
print_grid_header()
for i in market_data:
    print_grid_row(i)
    response = chain.invoke({
        "query": "Generate a twit",
        "market_data": i
    })
    print(f"Tweet: {response}")
    print("="*120)
    time.sleep(3)

def main():
    """Main function"""
    logger.info("🚀 Starting Twitter Context Generator v2.0...")
    
    try:
        # Initialize generator
        generator = TwitterContextGenerator()
        
        # Generate tweet
        result = generator.generate_tweet()
        
        # Display result
        print("\n🐦 Generated Tweet:")
        print(f"{result['tweet']}")
        print("\n📊 Context:")
        print(f"Tone: {result['context']['tone']}")
        print(f"Market: {result['context']['market_data']['crypto_market']['market_mood']}")
        print(f"Global Event: {result['context']['market_data']['global_events']['main_event']}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Generator stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        logger.info("👋 Generator finished")

if __name__ == "__main__":
    main() 