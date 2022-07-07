import pandas as pd
import json
from pathlib import Path


BASE_PATH = Path("/data/commonvoice/cv-corpus-9.0-2022-04-27")

LANG_CODE = "fr"


path = BASE_PATH / LANG_CODE


train = pd.read_csv(path / "train.tsv", sep="\t")
validated = pd.read_csv(path / "validated.tsv", sep="\t", low_memory=False)
dev = pd.read_csv(path / "dev.tsv", sep="\t", low_memory=False)
test = pd.read_csv(path / "test.tsv", sep="\t", low_memory=False)


print(f"{len(train)=} {len(validated)=}, {len(dev)=}, {len(test)=}")

ids = validated["sentence"]
validated[ids.isin(ids[ids.duplicated()])].sort_values("sentence")
print(
    f"duplicated sentences in validated.tsv: {len(validated[ids.isin(ids[ids.duplicated()])].sort_values('sentence'))}"
)
print(
    f"{validated[ids.isin(ids[ids.duplicated()])].sort_values('sentence').sentence.nunique()} unique sentences are duplicated"
)


df = pd.DataFrame.merge(validated, test, on="path", how="outer", indicator=True)
validated_train = df[df["_merge"] == "left_only"]
validated_train.columns = validated_train.columns.str.replace("sentence_x", "sentence")

SPLITE = ["dev", "test", "train", "validated_train"]

for split in SPLITE:

    # HACK
    target = eval(split)
    original_len = len(target)
    target = target.dropna(subset=["sentence"])
    pairs = []

    for _, row in target.iterrows():
        pair = {
            "file": row.path,
            "text": row.sentence,
        }

        pairs.append(pair)

    data = pairs
    print(f"{split}: {len(data)}")
    print(f"droped: {original_len - len(target)}")

    result = {"root": "/data/commonvoice/", "data": data}
    output = f"out-1/{LANG_CODE}-{split}.json"
    with open(f"{output}", "w") as f:
        json.dump(result, f)
