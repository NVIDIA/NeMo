import requests
import base64

# URL of the Gradio server
url = 'http://localhost:8890/api/predict/'

# Prepare the text data
text_data = '<image>Describe this image please.'

# Prepare the image data
with open("/path/to/images/001.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Data to send
data = {
    'data': [text_data, encoded_string]
}

# Sending a POST request to the Gradio server
response = requests.post(url, json=data)

# Checking if the request was successful
if response.status_code == 200:
    # Parsing the response
    response_data = response.json()
    print("Response from server:", response_data)
else:
    print("Failed to get a response from the server, status code:", response.status_code)