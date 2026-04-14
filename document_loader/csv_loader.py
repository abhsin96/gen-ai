from langchain_community.document_loaders import CSVLoader

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "Social_Network_Ads.csv")

loader = CSVLoader(file_path=file_path)

doc = loader.load()

print(doc[0])