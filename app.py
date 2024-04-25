from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-0125")

def classify_document(text):
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'Classification' function.

        Passage:
        {input}
        """
    )

    class Classification(BaseModel):
        category: str = Field(description="The category of the financial document.",
                              enum=['expenses', 'donations', 'income'])

    llmTagging = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
        Classification)

    tagging_chain = tagging_prompt | llmTagging

    return tagging_chain.invoke(input=text).category

def generate_prompt_template(category: str):
    # Define different prompt templates based on categories
    data_points = {
        "expenses": """
            "Date": "date of the expense",
            "Vendor": "vendor where the expense was made",
            "Amount": "amount of the expense",
            "Description": "description of the expense"
            """,

        "donations": """
            "Date": "date of the donation",
            "Organization": "organization receiving the donation",
            "Amount": "amount of the donation",
            "Description": "description of the donation"
            """,

        "income": """
            "Date": "date of the income",
            "Source": "source of the income",
            "Amount": "amount of the income",
            "Description": "description of the income"
            """
    }
    # Retrieve the appropriate prompt template based on the category
    prompt_template = data_points.get(category.lower(), "Unknown category")

    return prompt_template

def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert financial admin people who will extract core information from financial documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content and return in a JSON array format.
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = SequentialChain(
        classify_document,
        lambda category: generate_prompt_template(category),
        llm
    )

    results = chain.run(content)

    return results

# Example usage
content_text = "This is a financial document describing an expense."
data_points = "expenses"  # Specify the category for data extraction

extracted_details = extract_structured_data(content_text, data_points)
print("Extracted Details:", extracted_details)
