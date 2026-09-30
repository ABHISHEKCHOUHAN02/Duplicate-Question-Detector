import json

path = r"D:\vscode\ML\Duplicate Question Detector\data\finetuned_sbert\modules.json"
with open(path) as f:
    print(json.load(f))