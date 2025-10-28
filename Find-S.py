import pandas as pd

data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

print("Training Data:\n")
print(data)

positive_data = data[data['EnjoySport'] == 'Yes'].iloc[:, :-1].values

hypothesis = list(positive_data[0])
for example in positive_data:
    for i in range(len(hypothesis)):
        if example[i] != hypothesis[i]:
            hypothesis[i] = '?'
print("\nFinal Hypothesis:")
print(hypothesis)