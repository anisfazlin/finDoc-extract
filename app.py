from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
import json
import requests
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field

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

def extract_content(text: str):
    images_list = convert_pdf_to_images(text)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract

# 3. Categorization and Template Mapping Chain

class Classification(BaseModel):
    category: str = Field(description="The category of the financial document.", 
                          enum = ['expenses', 'donations', 'income'])

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

def process_document(text):

    detected_category = classify_document(text)
    print(detected_category)

    try:
        # Define the extraction prompt based on the detected category
        template_prompt = generate_prompt_template(detected_category)
        extracted_data = extract_structured_data(text, template_prompt)
        
        return extracted_data
    except:
        return "Detected category does not match the expected category."

def get_data(category: str, content: str):
    #Retrieve the appropriate prompt template based on the category
    data_points = generate_prompt_template(category)
    #Extract structured data
    data = extract_structured_data(content, data_points)
    return data

#4. Export to Google Sheets and Goold Drive
NGROK_URL = "https://b0dc-2402-1980-8240-7d4a-6917-6715-cbc3-9dfc.ngrok-free.app"

def authenticate_google():
    """Authenticate and return the Google API client."""
    creds = None
    if 'token.json' in os.listdir():
        creds = service_account.Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'secret_client.json'), SCOPES)
            flow.redirect_uri = f"{NGROK_URL}/callback"  # Set the redirect URI here
            creds = flow.run_local_server(port=8080)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

# # Function to search for an existing spreadsheet by name
# def search_spreadsheet(drive_service, spreadsheet_name):
#     query = f"name='{spreadsheet_name}' and mimeType='application/vnd.google-apps.spreadsheet'"
#     results = drive_service.files().list(q=query).execute()
#     files = results.get('files', [])
#     if files:
#         return files[0]['id']
#     return None

def create_or_update_spreadsheet(service, data, category, spreadsheet_name='Personal Ledger'):
    """Create or update a spreadsheet and return its ID."""
    try:
        # Check if the spreadsheet already exists
        drive_service = build('drive', 'v3', credentials=service)
        response = drive_service.files().list(q=f"name='{spreadsheet_name}'", spaces='drive').execute()
        files = response.get('files', [])

        if files:
            spreadsheet_id = files[0]['id']
            sheet_service = build('sheets', 'v4', credentials=service)
            sheet_metadata = sheet_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheet_names = [sheet['properties']['title'] for sheet in sheet_metadata['sheets']]
            
            if category in sheet_names:
                # Sheet exists, append data
                range_name = f"{category}!A1"
                response = sheet_service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": data}
                ).execute()
            else:
                # Create new sheet
                requests = [
                    {
                        "addSheet": {
                            "properties": {
                                "title": category
                            }
                        }
                    }
                ]
                body = {
                    'requests': requests
                }
                response = sheet_service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body=body
                ).execute()

                # Add data to new sheet
                range_name = f"{category}!A1"
                response = sheet_service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body={"values": data}
                ).execute()

            return spreadsheet_id
        else:
            # Create new spreadsheet
            sheet_service = build('sheets', 'v4', credentials=service)
            spreadsheet = {
                'properties': {
                    'title': spreadsheet_name
                },
                'sheets': [
                    {
                        'properties': {
                            'title': category
                        }
                    }
                ]
            }
            spreadsheet = sheet_service.spreadsheets().create(body=spreadsheet).execute()
            spreadsheet_id = spreadsheet.get('spreadsheetId')

            # Add data to the newly created sheet
            range_name = f"{category}!A1"
            response = sheet_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                body={"values": data}
            ).execute()

            return spreadsheet_id
    except HttpError as err:
        st.error(f"An error occurred: {err}")
        return None
    
# def main():
#     st.markdown('Generate your personal ledger by uploading your financial documents - powered by Artificial Intelligence.')
#     st.title("FiscalFile Forge: Unleash your financial data")
    
#     st.write('\n')  # add spacing

#     st.subheader('\nWhat is your email all about?\n')
#     with st.expander("SECTION - Upload Financial Documents", expanded=True):
#         uploaded_files = st.file_uploader("Choose an image financial file", accept_multiple_files=True)
        
#         if uploaded_files:
#             results = []  # Initialize results list outside the loop

#             # Process each uploaded file
#             for uploaded_file in uploaded_files:
#                 with NamedTemporaryFile(suffix='.csv') as f:
#                     f.write(uploaded_file.getbuffer())
#                     f.seek(0)  # Reset file pointer to beginning
#                     content = extract_content(f.name)
                    
#                     # Classify document and extract structured data
#                     # category = classify_document(content)
#                     # data = process_document(content)
#                     classification_result = classify_document(content)
#                     category = classification_result
                    
#                     categories = ["expenses", "donations", "income"]
#                     #DYnamically change displayed category based on classification
#                     selected_category = st.selectbox("Select a category to view the data", categories, index=categories.index(category))
                    
#                     data_points_template = generate_prompt_template(selected_category)
#                     data = extract_structured_data(content, data_points_template)
                    
#                     # Parse data and append to results
#                     json_data = json.loads(data)
#                     if isinstance(json_data, list):
#                         results.extend(json_data)  # Use extend() for lists
#                     else:
#                         results.append(json_data)  # Wrap the dict in a list

#             # Display results if any documents were processed
#             if results:
#                 try:
#                     df = pd.DataFrame(results)
#                     st.subheader("Results")
                    
#                     st.data_editor(df)

#                 except Exception as e:
#                     st.error(f"An error occurred while creating the DataFrame: {e}")
#                     st.write(results)  # Print the data to see its content

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     main()

def main():
    st.markdown('Generate your personal ledger by uploading your financial documents - powered by Artificial Intelligence.')
    st.title("FiscalFile Forge: Unleash your financial data")
    
    st.write('\n')  # add spacing

    st.subheader('\nWhat is your email all about?\n')
    with st.expander("SECTION - Upload Financial Documents", expanded=True):
        uploaded_files = st.file_uploader("Choose an image financial file", accept_multiple_files=True)
        
        if uploaded_files:
            results = []  # Initialize results list outside the loop

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                with NamedTemporaryFile(suffix='.csv') as f:
                    f.write(uploaded_file.getbuffer())
                    f.seek(0)  # Reset file pointer to beginning
                    content = extract_content(f.name)
                    
                    # Classify document and extract structured data
                    # category = classify_document(content)
                    # data = process_document(content)
                    classification_result = classify_document(content)
                    category = classification_result
                    
                    categories = ["expenses", "donations", "income"]
                    #DYnamically change displayed category based on classification
                    selected_category = st.selectbox("Select a category to view the data", categories, index=categories.index(category))
                    
                    data_points_template = generate_prompt_template(selected_category)
                    data = extract_structured_data(content, data_points_template)
                    
                    # Parse data and append to results
                    json_data = json.loads(data)
                    if isinstance(json_data, list):
                        results.extend(json_data)  # Use extend() for lists
                    else:
                        results.append(json_data)  # Wrap the dict in a list

            # Display results if any documents were processed
            if results:
                try:
                    df = pd.DataFrame(results)
                    st.subheader("Results")
                    
                    st.data_editor(df)

                except Exception as e:
                    st.error(f"An error occurred while creating the DataFrame: {e}")
                    st.write(results)  # Print the data to see its content

            # Export to Google Sheets Button
                if st.button("Export to Google Sheets"):
                    creds = authenticate_google()
                    spreadsheet_id = create_or_update_spreadsheet(creds, df.values.tolist(), selected_category)
                    if spreadsheet_id:
                        st.success(f'Data successfully exported to Google Sheets. [Click here to open the spreadsheet](https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit)')

if __name__ == '__main__':
    main()
