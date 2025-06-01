import json
import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key="xxxx")

# Load the data
df = pd.read_csv("data/transcriptions.csv")
df.head()

# Define function to extract age and recommended treatment/procedure
def extract_info_with_openai(transcription):
    """Extracts age and recommended treatment from a transcription using OpenAI."""
    messages = [
        {
            "role": "system",
            "content": "You are a healthcare professional extracting patient data. Always return both the age and recommended treatment. If the information is missing, still create the field and specify 'Unknown'.",
            "role": "user",
            "content": f"Please extract and return both the patient's age and recommended treatment from the following transcription. Transcription: {transcription}."
        }
    ]
    function_definition = [
        {
            'type': 'function',
            'function': {
                'name': 'extract_medical_data',
                'description': 'Get the age and recommended treatment from the input text. Always return both age and recommended treatment.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'Age': {
                            'type': 'integer',
                            'description': 'Age of the patient'
                        },
                        'Recommended Treatment/Procedure': {
                            'type': 'string',
                            'description': 'Recommended treatment or procedure for the patient'
                        }
                    }
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=function_definition
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

# Define function to extract age and recommended treatment/procedure
def get_icd_codes(treatment):
    if treatment != 'Unknown':
        """Retrieves ICD codes for a given treatment using OpenAI."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Provide the ICD codes for the following treatment or procedure: {treatment}. Return the answer as a list of codes. Please only include the codes and no other information."
            }],
            temperature=0.3
        )
        output = response.choices[0].message.content
    else:
        output = 'Unknown'
    return output

# Start an empty list to store processed data
processed_data = []

# Process each row in the DataFrame
for index, row in df.iterrows():
    medical_specialty = row['medical_specialty']
    extracted_data = extract_info_with_openai(row['transcription'])
    icd_code = get_icd_codes(extracted_data["Recommended Treatment/Procedure"]) if 'Recommended Treatment/Procedure' in extracted_data.keys() else 'Unknown'
    extracted_data["Medical Specialty"] = medical_specialty
    extracted_data["ICD Code"] = icd_code

    # Append the extracted information as a new row in the list
    processed_data.append(extracted_data)

# Convert the list to a DataFrame
df_structured = pd.DataFrame(processed_data)