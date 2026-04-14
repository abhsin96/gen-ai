"""WebBaseLoader Example - Scraping and Processing Web Content

This script demonstrates how to:
1. Load and scrape web content using WebBaseLoader
2. Set USER_AGENT to identify requests properly
3. Process scraped content using LangChain with optimized prompts
4. Extract structured information from web pages
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import threading
import time
import sys

# Load environment variables
load_dotenv()

# Target URL for scraping
url = "https://www.carwale.com/mahindra-cars/xuv-3xo/"

# Initialize WebBaseLoader with the URL
loader = WebBaseLoader(url)

# Load the web content
print("Loading web content...")
doc = loader.load()
print(f"Successfully loaded {len(doc)} document(s)\n")

# Initialize the language model
model = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize output parser
parser = StrOutputParser()

# Optimized prompt using ChatPromptTemplate for better structure and clarity
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automotive expert assistant specializing in analyzing vehicle specifications and variants. Your task is to extract and present information in a clear, structured format."),
    ("human", """Analyze the following web content about the Mahindra XUV 3XO and provide a comprehensive summary of all available models.

For each model variant, extract and organize:
1. Model/Variant Name
2. Engine Specifications (displacement, type, power, torque)
3. Transmission Options (manual/automatic, number of gears)
4. Key Features or Highlights

Present the information in a well-structured, easy-to-read format.

Web Content:
{docs}

Provide a detailed analysis:""")
])

# Create the processing chain using LCEL (LangChain Expression Language)
chain = prompt | model | parser

# Loading animation function
def loading_animation(stop_event):
    """Display animated loading dots while processing"""
    animation = [".", "..", "...", "...."]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rProcessing web content with AI{animation[idx % len(animation)]}   ")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.3)
    sys.stdout.write("\rProcessing web content with AI... Done!\n")
    sys.stdout.flush()

# Execute the chain with the scraped content and loading animation
stop_animation = threading.Event()
animation_thread = threading.Thread(target=loading_animation, args=(stop_animation,))
animation_thread.start()

try:
    result = chain.invoke({
        "docs": doc[0].page_content
    })
finally:
    stop_animation.set()
    animation_thread.join()
    print()  # Add newline after animation

# Display the results
print("=" * 80)
print("MAHINDRA XUV 3XO - MODEL VARIANTS ANALYSIS")
print("=" * 80)
print(result)
print("=" * 80)