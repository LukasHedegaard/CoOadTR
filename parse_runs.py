from pathlib import Path
import re


def read_mAPs(filename: str):
    pattern = re.compile(r"mAP: ([\d.]+)")
    mAPs = []
    for line in open(filename):
        for match in re.finditer(pattern, line):
            mAPs.append(float(match.group(1)))
    return mAPs


def name2vars(filename):
    pattern = re.compile(r"oadtr-(\w\d)_clspos([-\d]+)_seed(\d)_([-\w]+).txt")
    block, clspos, seed, dataset = re.match(pattern, filename).groups()
    clspos = {
        "-1": "-",
        "0": "1A",
        "1": "1F",
        "2": "2A",
        "3": "2F",
        "4": "3A",
    }[clspos]
    return (block, clspos, seed, dataset)


run_files = (Path(__file__).parent / "runs").glob("*.txt")
results = []

for f in run_files:
    block, clspos, seed, dataset = name2vars(f.name)
    mAPs = read_mAPs(f)
    results.append([block, clspos, seed, dataset, mAPs])


print("epoch1_data=", [[*r[:-1], r[-1][0]] for r in results])

print("\nepoch2_data=", [[*r[:-1], r[-1][1]] for r in results])

print("\nepoch3_data=", [[*r[:-1], r[-1][2]] for r in results])

print("\nepoch4_data=", [[*r[:-1], r[-1][3]] for r in results])

print("\nepoch_best_data=", [[*r[:-1], max(r[-1])] for r in results])
