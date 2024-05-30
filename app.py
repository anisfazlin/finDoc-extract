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
from langchain_core.pydantic_v1 import BaseModel, Field
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-0125")

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

#CSS Style for Streamlit page
#Remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)


# 1. Convert PDF file into images via pypdfium2

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

# 2. Extract text from images via pytesseract

def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_content(file_path, file_extension):
    if file_extension == 'pdf':
        images_list = convert_pdf_to_images(file_path)
        if images_list is None:
            return None
        text_with_pytesseract = extract_text_from_img(images_list)
    else:
        # Handle image files directly
        with open(file_path, 'rb') as image_file:
            image = Image.open(image_file)
            text_with_pytesseract = image_to_string(image)
    return text_with_pytesseract
    
    
# def convert_pdf_to_images(file_path, scale=300/72):
#     try:
#         pdf_file = pdfium.PdfDocument(file_path)
#         page_indices = [i for i in range(len(pdf_file))]
#         renderer = pdf_file.render(
#             pdfium.PdfBitmap.to_pil,
#             page_indices=page_indices,
#             scale=scale
#         )
#         final_images = []

#         for i, image in zip(page_indices, renderer):
#             image_byte_array = BytesIO()
#             image.save(image_byte_array, format='jpeg', optimize=True)
#             image_byte_array = image_byte_array.getvalue()
#             final_images.append(dict({i: image_byte_array}))

#         return final_images
#     except pdfium.PdfiumError as e:
#         st.error(f"Failed to load PDF document: {e}")
#         return None

# 2. Extract text from images via pytesseract

# def extract_text_from_img(list_dict_final_images):
#     image_list = [list(data.values())[0] for data in list_dict_final_images]
#     image_content = []

#     for index, image_bytes in enumerate(image_list):
#         image = Image.open(BytesIO(image_bytes))
#         raw_text = str(image_to_string(image))
#         image_content.append(raw_text)

#     return "\n".join(image_content)


# def extract_content(file_path, file_type):
#     if file_type == 'pdf':
#         images_list = convert_pdf_to_images(file_path)
#         if images_list is None:
#             return None
#         text_with_pytesseract = extract_text_from_img(images_list)
#         return text_with_pytesseract
#     else:
#         image = Image.open(file_path)
#         text_with_pytesseract = image_to_string(image)
#         return text_with_pytesseract

# def extract_content(file: str):
#     images_list = convert_pdf_to_images(file)
#     text_with_pytesseract = extract_text_from_img(images_list)
#     return text_with_pytesseract
    # file_extension = os.path.splitext(file)[1].lower()
    # print(file_extension)
    
    # if file_extension == ".pdf":
    #     images_list = convert_pdf_to_images(file)
    #     text_with_pytesseract = extract_text_from_img(images_list)
    # elif file_extension in [".jpg", ".jpeg", ".png"]:
    #     text_with_pytesseract = extract_text_from_image(file)
    # else:
    #     raise ValueError(f"Unsupported file type: {file_extension}")

    # return text_with_pytesseract
    
# 3. Categorization and Template Mapping Chain

class Classification(BaseModel):
    category: str = Field(description="The category of the financial document.", 
                        #   enum = ['expenses', 'donations', 'income'])
                        enum = ['bank statement', 'invoice', 'receipt', 'pay slip'])

def classify_document(text):
    
    tagging_prompt = PromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
        
    llmTagging = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification)

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
        "pay slip": {
            "Employee Name": "Name of the employee.",
            "Employee ID": "Unique identifier for the employee.",
            "Employer Name": "Name of the employer.",
            "Pay Period": "The date range for which the pay slip applies.",
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
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)
    return results

#4. Export to Google Sheets

def authenticate_google():
    """Authenticate and return the Google API client."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'secret_client.json', SCOPES)
            flow.redirect_uri = f"{NGROK_URL}/callback"  # Set the redirect URI here
            creds = flow.run_local_server(port=8080)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds


def create_or_update_spreadsheet_with_multiple_sheets(creds, all_data):
    try:
        # Check for existing spreadsheet
        spreadsheet_id = "YOUR_SPREADSHEET_ID"  # Replace with the actual ID or a mechanism to retrieve it
        service = build('sheets', 'v4', credentials=creds)

        # If the spreadsheet exists, update it
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

        for category, data in all_data.items():
            df = pd.DataFrame(data)
            headers = df.columns.tolist()
            values = [headers] + df.astype(str).values.tolist()

            body = {
                "values": values
            }

            # Check if the sheet already exists
            sheet_exists = False
            for sheet in spreadsheet.get('sheets', []):
                if sheet.get("properties", {}).get("title") == category:
                    sheet_exists = True
                    break

            if sheet_exists:
                # Update existing sheet
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{category}!A1",
                    valueInputOption="RAW",
                    body=body
                ).execute()
            else:
                # Add new sheet
                requests = [{
                    "addSheet": {
                        "properties": {
                            "title": category
                        }
                    }
                }]
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"requests": requests}
                ).execute()

                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{category}!A1",
                    valueInputOption="RAW",
                    body=body
                ).execute()

        return spreadsheet_id

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


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
            headers = list(data[0].keys())
            values = [headers] + [[str(item.get(header, '')) for header in headers] for item in data]

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

def main():
    st.markdown('Generate your personal ledger by uploading your financial documents - powered by Artificial Intelligence.')
    st.title("FinData - Unstructured Data to Structured Financial Data Extraction & Classification")

    st.write('\n')  # add spacing

    st.subheader('\nWhat is your financial data is all about?\n')
    with st.expander("SECTION - Upload Financial Documents", expanded=True):
        uploaded_files = st.file_uploader("Upload PDF or images financial data", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])

        if uploaded_files:
            results = {}  # Initialize results dictionary to store data per category

            for idx, uploaded_file in enumerate(uploaded_files):
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
                
                # st.write(content)

                classification_result = classify_document(content)
                category = classification_result

                data_points_template = generate_prompt_template(category)
                data = extract_structured_data(content, data_points_template)

                json_data = json.loads(data)
                if isinstance(json_data, list):
                    if category in results:
                        results[category].extend(json_data)
                    else:
                        results[category] = json_data
                else:
                    if category in results:
                        results[category].append(json_data)
                    else:
                        results[category] = [json_data]

            # Display results if any documents were processed
            if results:
                for category, data in results.items():
                    try:
                        # Check if the first item in the list is a dictionary, and get keys from it
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            df = pd.DataFrame(data)
                        else:
                            st.error(f"The data for {category} is not in the expected format.")
                            continue

                        st.subheader(f"Results for {category.capitalize()}")
                        st.data_editor(df, key=f"data_editor_{category}")

                        if st.button(f"Export {category} to Google Sheets", key=f"export_button_{category}"):
                            creds = authenticate_google()
                            if creds:
                                spreadsheet_id = create_or_update_spreadsheet_with_multiple_sheets(creds, {category: data})
                                if spreadsheet_id:
                                    st.success(f'Data successfully exported to Google Sheets. [Click here to open the {category.capitalize()} spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit)')

                    except Exception as e:
                        st.error(f"An error occurred while creating the DataFrame for {category}: {e}")
                        st.write(data)

                # Add an "Export All" button
                if st.button("Export All Data to Google Sheets"):
                    creds = authenticate_google()
                    if creds:
                        spreadsheet_id = create_or_update_spreadsheet_with_multiple_sheets(creds, results)
                        if spreadsheet_id:
                            st.success(f'All data successfully exported to Google Sheets. [Click here to open the spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit)')

if __name__ == '__main__':
    main()
