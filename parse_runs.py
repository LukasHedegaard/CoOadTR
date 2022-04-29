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
    pattern = re.compile(r"oadtr-(\w\d)_clspos([-\d]+)_seed(\d)")
    block, clspos, seed = re.match(pattern, filename).groups()
    clspos = {
        "-1": "-",
        "0": "1A",
        "1": "1F",
        "2": "2A",
        "3": "2F",
        "4": "3A",
    }[clspos]
    return (block, clspos, seed)


run_files = (Path(__file__).parent / "runs").glob("*.txt")
results = []

for f in run_files:
    block, clspos, seed = name2vars(f.name)
    mAPs = read_mAPs(f)
    results.append([block, clspos, seed, mAPs])


print("epoch 1 results:")
print([[*r[:-1], r[-1][0]] for r in results])

print("\nepoch 2 results:")
print([[*r[:-1], r[-1][1]] for r in results])

print("\nepoch 3 results:")
print([[*r[:-1], r[-1][2]] for r in results])

print("\nepoch 4 results:")
print([[*r[:-1], r[-1][3]] for r in results])

print("\nbest results:")
print([[*r[:-1], max(r[-1])] for r in results])
