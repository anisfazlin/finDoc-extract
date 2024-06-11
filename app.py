from pytesseract import image_to_string
from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
from demjson3 import decode
import cv2
import numpy as np
import pandas as pd
import re
import json
import requests
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

open_api_key = st.secrets["OPENAI_API_KEY"]
# ngrok_url = st.secrets["ngrok_url"]


llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-0125")
# llm = ChatGroq()

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

#CSS Style for Streamlit page
#Remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)

#1. Data Preprocessing for OCR

# 2. Convert PDF file into images via pypdfium2

def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale
    )
    final_images = []

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

# # 3. Extract text from images via pytesseract

def extract_text_from_img(list_dict_final_images, whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$@*"):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    custom_config = f'-c tessedit_char_whitelist={whitelist}'
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image, config = custom_config))
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_content(file_path, file_extension, whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$@*"):
    custom_config = f'-c tessedit_char_whitelist={whitelist}'
    if file_extension == 'pdf':
        images_list = convert_pdf_to_images(file_path)
        if images_list is None:
            return None
        text_with_pytesseract = extract_text_from_img(images_list)
    else:
        # Handle image files directly
        with open(file_path, 'rb') as image_file:
            image = Image.open(image_file)
            text_with_pytesseract = image_to_string(image, config=custom_config)
    return text_with_pytesseract

def parse_json_robustly(content):
    try:
        data = decode(content)
        return data
    except Exception as e:  # Catch any parsing errors (including demjson exceptions)
        print(f"Error parsing JSON: {e}")
        return None
    
# 4. Categorization and Template Mapping Chain

class Classification(BaseModel):
    category: str = Field(description="The category of the financial document.", 
                        #   enum = ['expenses', 'donations', 'income'])
                        enum = ['bank statement', 'invoice', 'receipt', 'income statement'])

def classify_document(text):
    
    tagging_prompt = PromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
        
    llmTagging = ChatOpenAI(model="gpt-3.5-turbo-0125").with_structured_output(Classification)
    # tagging_chain = llm.with_structured_output(Classification)

    tagging_chain = tagging_prompt | llmTagging    
    
    return tagging_chain.invoke(input=text).category

def generate_prompt_template(category: str):
    data_points = {
        "bank statement": {
            "Bank Name": "Name of the bank.",
            "Statement Period": "The date range for the statement.",
            "Transaction Date": "Date of each transaction.",
            "Transaction Description": "Description of each transaction.",
            "Transaction Amount": "Amount of money involved in each transaction.",
            "Transaction Category/Type": "Category of the transaction (e.g., groceries, utilities).",
            "Amount Due": "Total amount due.",
            "Balance": "Account balance after each transaction.",
            "Interest Earned": "Any interest earned during the period.",
            "Fees and Charges": "Details of any fees or charges applied."
        },
        "invoice": {
            "Invoice Number": "Unique identifier for the invoice.",
            "Invoice Date": "Date the invoice was issued.",
            "Vendor Name": "Name of the vendor.",
            "Vendor Address": "Address of the vendor.",
            "Customer Name": "Name of the customer.",
            "Description": "Description of each line item or overall invoice.",
            "Taxes": "Amount of taxes applied.",
            "Discounts": "Any discounts applied.",
            "Total Amount Due": "Total amount payable.",
            "Category/Type": "Category of the expense (e.g., utilities, rent, donation etc)."
        },
        "receipt": {
            "Store Name": "Name of the store.",
            "Store Address": "Address of the store.",
            "Date of Purchase": "Date when the purchase was made.",
            "Time of Purchase": "Time of the purchase.",
            "Transaction ID": "Unique identifier for the transaction.",
            "Descriptions": "Description like Items bought, and quantities.",
            "Total Amount": "Total amount paid.",
            "Payment Method": "Method of payment used (e.g., credit card, cash).",
            "Category/Type": "Category of the purchase (e.g., groceries, dining, foood&beverage etc)."
        },
        "income statement": {
            "Employee Name": "Name of the employee.",
            "Employee ID": "Unique identifier for the employee.",
            "Employer Name": "Name of the employer.",
            "Pay Period": "The date range for which the income statement applies.",
            "Gross Salary": "Total earnings before deductions.",
            "Net Salary": "Total earnings after deductions.",
            "Deductions": "Breakdown of deductions (e.g., taxes, insurance, retirement contributions).",
            "Bonuses": "Any bonuses or additional compensation.",
            "Tax Information": "Details of taxes deducted."
        }
    }

    # Retrieve the appropriate prompt template based on the category
    prompt_template = data_points.get(category.lower(), "Unknown category")

    return prompt_template

def extract_structured_data(content: str, data_points):

    template = """
    You are an expert financial admin people who will extract core information from financial documents

    {content}

    Above is the content; please try to extract all data points and return in a JSON array format
    {data_points}
    
    If there is no data available, please return an JSON array with 'N/A' values. Strictly only display JSON array format
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)
    return results

#5. Export to Google Sheets

def authenticate_google():
    """Authenticate and return the Google API client."""
    creds = None
    token_path = 'token.json'
    
    # Load credentials from token.json if it exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load client secrets from Streamlit secrets
            client_config = {
                "installed": {
                    "client_id": st.secrets["google_client_id"],
                    "client_secret": st.secrets["google_client_secret"],
                    "redirect_uris": [st.secrets["google_redirect_uri"]],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            flow.redirect_uri = st.secrets["google_redirect_uri"]
            creds = flow.run_local_server(port=8080)
        
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return creds

def get_spreadsheet_by_title(creds, title):
    try:
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(q=f"name='{title}' and mimeType='application/vnd.google-apps.spreadsheet'",
                                       fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            return None
        return items[0]['id']
    except HttpError as error:
        st.error(f"An error occurred: {error}")
        return None

def create_or_update_spreadsheet_with_multiple_sheets(creds, all_data, spreadsheet_name='Personal Ledger'):
    try:
        service = build('sheets', 'v4', credentials=creds)
        spreadsheet_id = get_spreadsheet_by_title(creds, spreadsheet_name)

        if not spreadsheet_id:
            # Create new spreadsheet if not exists
            spreadsheet = {
                'properties': {
                    'title': spreadsheet_name
                },
                'sheets': [{'properties': {'title': category}} for category in all_data.keys()]
            }
            spreadsheet = service.spreadsheets().create(body=spreadsheet).execute()
            spreadsheet_id = spreadsheet['spreadsheetId']
        else:
            # Add new sheets to existing spreadsheet
            existing_sheets = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute().get('sheets', [])
            existing_sheet_titles = [sheet['properties']['title'] for sheet in existing_sheets]

            new_sheets = [{'properties': {'title': category}} for category in all_data.keys() if category not in existing_sheet_titles]
            if new_sheets:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={'requests': [{'addSheet': sheet} for sheet in new_sheets]}
                ).execute()

        for category, data in all_data.items():
            if not data:
                continue

            df = pd.DataFrame(data)

            # Convert columns to appropriate data types
            # for column in df.columns:
            #     if column.lower() in ["date", "transaction date", "invoice date", "pay period", "date of purchase", "statement period"]:
            #         df[column] = pd.to_datetime(df[column], errors='coerce')  # Convert to datetime
            #     elif column.lower() in ["amount", "balance", "total amount", "gross salary", "net salary", "transaction amount", "interest earned", "fees and charges"]:
            #         df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric

            headers = df.columns.tolist()
            values = [headers] + df.astype(str).values.tolist()

            body = {
                'values': values
            }
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"{category}!A1",
                valueInputOption='RAW',
                body=body
            ).execute()

        return spreadsheet_id
    except HttpError as error:
        st.error(f"An error occurred: {error}")
        return None

def standardize_json(data):
    def normalize_key(key):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()

    if isinstance(data, dict):
        return {normalize_key(k): standardize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [standardize_json(item) for item in data]
    else:
        return data

def validate_json(content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.text(f"Content causing error: {content[:500]}")
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)
        content = re.sub(r"'", '"', content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            st.error(f"Error after attempting to fix common issues: {e}")
            st.text(f"Content causing error: {content[:500]}")
            return None
    return data

def main():
    st.markdown('Generate your personal ledger by uploading your financial documents - powered by Artificial Intelligence.')
    st.title("FinData - Unstructured Data to Structured Financial Data Extraction & Classification")

    st.write('\n')  # add spacing

    st.subheader('\nWhat is your financial data is all about?\n')
    with st.expander("SECTION - Upload Financial Documents", expanded=True):
        uploaded_files = st.file_uploader("Upload PDF or images financial data", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])

        if uploaded_files:
            if "results" not in st.session_state:
                st.session_state.results = {}  # Initialize results dictionary to store data per category

            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()  # Set to keep track of processed files

            if "selected_categories" not in st.session_state:
                st.session_state.selected_categories = {}  # Initialize selected categories for each file

            for idx, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.name in st.session_state.processed_files:
                    # st.info(f"File {uploaded_file.name} has already been processed.")
                    continue

                file_extension = uploaded_file.name.split('.')[-1].lower()
                suffix = f".{file_extension}"

                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                content = extract_content(temp_file_path, file_extension)
                os.remove(temp_file_path)

                if content is None:
                    st.error(f"Could not process the file: {uploaded_file.name}")
                    continue

                if uploaded_file.name not in st.session_state.selected_categories:
                    with st.spinner("Extracting structured data..."):
                        classification_result = classify_document(content)
                        category = classification_result
                        st.session_state.selected_categories[uploaded_file.name] = category
                else:
                    category = st.session_state.selected_categories[uploaded_file.name]

                categories = ["bank statement", "invoice", "receipt", "income statement"]
                selected_category = st.selectbox(f"Select a category for file {uploaded_file.name}", categories, index=categories.index(category), key=f"selectbox_{uploaded_file.name}")

                st.session_state.selected_categories[uploaded_file.name] = selected_category

                data_points_template = generate_prompt_template(selected_category)
                data = extract_structured_data(content, data_points_template)

                try:
                    json_data = validate_json(data)
                    if json_data is None:
                        json_data = parse_json_robustly(content)
                except Exception as e:
                    st.error(f"Could not process JSON for {uploaded_file.name}: {e}")
                    st.write(f"Raw content:\n{data}")
                    continue

                if isinstance(json_data, list):
                    if selected_category in st.session_state.results:
                        st.session_state.results[selected_category].extend(json_data)
                    else:
                        st.session_state.results[selected_category] = json_data
                else:
                    if selected_category in st.session_state.results:
                        st.session_state.results[selected_category].append(json_data)
                    else:
                        st.session_state.results[selected_category] = [json_data]

                st.session_state.processed_files.add(uploaded_file.name)

            if st.session_state.results:
                for selected_category, data in st.session_state.results.items():
                    try:
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            df = pd.DataFrame(data)
                            for col in df.columns:
                                if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
                                    df[col] = df[col].astype(str)
                        else:
                            st.error(f"The data for {selected_category} is not in the expected format.")
                            continue

                        st.write('\n')
                        st.subheader(f"Results for {selected_category.capitalize()}")
                        edited_df = st.data_editor(df, key=f"data_editor_{selected_category}")

                        st.session_state.results[selected_category] = edited_df.to_dict(orient='records')

                        if st.button(f"Export {selected_category} to Google Sheets", key=f"export_button_{selected_category}"):
                            creds = authenticate_google()
                            if creds:
                                spreadsheet_id = create_or_update_spreadsheet_with_multiple_sheets(creds, {selected_category: st.session_state.results[selected_category]})
                                if spreadsheet_id:
                                    st.success(f'Data successfully exported to Google Sheets. [Click here to open the {selected_category.capitalize()} spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit)')

                    except Exception as e:
                        st.error(f"An error occurred while creating the DataFrame for {selected_category}: {e}")
                        st.write(data)

                st.write('\n')
                if st.button("Export All Data to Google Sheets"):
                    creds = authenticate_google()
                    if creds:
                        spreadsheet_id = create_or_update_spreadsheet_with_multiple_sheets(creds, st.session_state.results)
                        if spreadsheet_id:
                            st.success(f'All data successfully exported to Google Sheets. [Click here to open the spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit)')                    
if __name__ == '__main__':
    main()

