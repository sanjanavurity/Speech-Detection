import pandas as pd

data = {
    "text": [
        "You are such a disgusting person",
        "Shut up already!",
        "I love this beautiful day",
        "You people are all the same",
        "Don't be so rude to others",
        "This app is amazing"
    ],
    "label": [0, 1, 2, 0, 1, 2]
}

df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)
print("test_data.csv created successfully!")
