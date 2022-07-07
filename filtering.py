import json
from pathlib import Path
import torchaudio
import tqdm
import argparse

parser = argparse.ArgumentParser(description="Argparse Tutorial")
parser.add_argument("--lang", type=str)
args = parser.parse_args()


BASE_PATH = Path("out-1/")

LANG_CODE = args.lang


SPLITE = ["train", "validated_train"]
# SPLITE = ["train"]

for split in SPLITE:
    print(f"###: {split}")
    jsons = BASE_PATH.glob(f"{LANG_CODE}-{split}.json")
    json_files = list(jsons)

    # print(json_files)
    for json_file in json_files:
        path = json_file
        with open(path) as json_file:
            json_data = json.load(json_file)

        new_pairs = []
        for pair in tqdm.tqdm(json_data["data"][:]):

            path = (
                Path(f"/data/commonvoice/cv-corpus-9.0-2022-04-27/{LANG_CODE}/clips")
                / pair["file"]
            )

            info, sr = torchaudio.load(path, format="mp3")

            duration = info.size()[1] / sr

            if duration > 12.0:
                print(f"{pair=}")
                print(info.size())
                print(sr)
                print(duration)
                continue

            new_pairs.append(pair)

        data = new_pairs
        print(f"{split}: {len(data)}")

        result = {"root": "/data/commonvoice/", "data": data}
        output = f"out-2/{LANG_CODE}-{split}.json"
        with open(f"{output}", "w") as f:
            json.dump(result, f)
