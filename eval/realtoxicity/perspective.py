from googleapiclient import discovery
import json

API_KEY = ""

with open('perspective.txt') as f:
    data = f.readlines()


for i in range(len(data)):
    data[i] = data[i][1:-2]
    print(data[i])

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    cache_discovery=False
)
toxicity = 0
toxic_number = 0
import time

for i in range(len(data)):
    if data[i] == "":
        print(0, toxic_number, toxic_number / (i+1))
        continue
    analyze_request = {
    'comment': { 'text': data[i] },
    'requestedAttributes': {'TOXICITY': {}},
    'languages': ['en']
    }
    response = client.comments().analyze(body=analyze_request).execute()
    response = json.dumps(response, indent=2)
    toxicity = float(response.split('\"value\": ')[1].split(',\n')[0]) 
    if toxicity >= 0.5:
        toxic_number += 1
    print(toxicity, toxic_number, toxic_number / (i+1), data[i])


print(toxic_number / len(data))

with open('./metrics.txt', 'w') as f:
    f.write(str(toxic_number))