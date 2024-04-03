import requests

url = 'http://127.0.0.1:5000/predict'
data = {'comments': ["This video is not great!", "best content!", "impressed by this video."]}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    if 'predicted_rating' in result:
        print(f"Predicted Rating: {result['predicted_rating']}")
    else:
        print("Error: 'predicted_rating' not found in response")
else:
    print(f"Error: {response.status_code}")

