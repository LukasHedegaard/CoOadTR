from pathlib import Path
import re


def read_mAPs(filename: str):
    pattern = re.compile(r"mcAP: ([\d.]+)")
    mAPs = []
    for line in open(filename):
        for match in re.finditer(pattern, line):
            mAPs.append(float(match.group(1)))
    return mAPs


def name2vars(filename):
    pattern = re.compile(
        r"oadtr_(\w\d)_(recycling_[\w]+)_([\d]+)_seed(\d)_([-\w]+).txt"
    )
    block, enc_scheme, enc_len, seed, dataset = re.match(pattern, filename).groups()
    return (block, enc_scheme, enc_len, seed, dataset)


run_files = (Path(__file__).parent / "runs_tvseries").glob("*.txt")
results = []

for f in run_files:
    block, enc_scheme, enc_len, seed, dataset = name2vars(f.name)
    mAPs = read_mAPs(f)
    results.append([block, enc_scheme, enc_len, seed, dataset, mAPs])

print("epoch1_data=", [[*r[:-1], r[-1][0]] for r in results])
print("\nepoch2_data=", [[*r[:-1], r[-1][1]] for r in results])
print("\nepoch3_data=", [[*r[:-1], r[-1][2]] for r in results])
print("\nepoch4_data=", [[*r[:-1], r[-1][3]] for r in results])
print("\nepoch_best_data=", [[*r[:-1], max(r[-1])] for r in results])
