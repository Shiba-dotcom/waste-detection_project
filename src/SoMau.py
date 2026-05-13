import json
import pandas as pd
from collections import Counter
with open("../data/raw/annotations.json", "r") as f:
    data = json.load(f)

mapping_df = pd.read_csv("mapping.csv")
mapping = dict(zip(mapping_df["name"], mapping_df["group"]))

idtn = {c["id"]: c["name"] for c in data["categories"]}

labels = []
for ann in data["annotations"]:
    name = idtn[ann["category_id"]]
    labels.append(mapping.get(name, "Other"))
    
counts = Counter(labels)
    
df_5 = pd.DataFrame(counts.items(), columns=["Class", "Count"])
df_5 = df_5.sort_values("Count", ascending=False)

df_5.to_csv("So_Mau.csv", index=False)