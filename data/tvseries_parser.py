from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

NUM_CLASSES = 31  # index 0 is the "None-class"

VIDEO_LENGTHS = {  # at 5 fps
    "24_ep1": 12290,
    "24_ep2": 12385,
    "24_ep3": 12685,
    "24_ep4": 12465,
    "Breaking_Bad_ep1": 16720,
    "Breaking_Bad_ep2": 13875,
    "Breaking_Bad_ep3": 13870,
    "How_I_Met_Your_Mother_ep1": 6348,
    "How_I_Met_Your_Mother_ep2": 6333,
    "How_I_Met_Your_Mother_ep3": 6140,
    "How_I_Met_Your_Mother_ep4": 6335,
    "How_I_Met_Your_Mother_ep5": 6334,
    "How_I_Met_Your_Mother_ep6": 6360,
    "How_I_Met_Your_Mother_ep7": 6340,
    "How_I_Met_Your_Mother_ep8": 6340,
    "Mad_Men_ep1": 14590,
    "Mad_Men_ep2": 14105,
    "Mad_Men_ep3": 13260,
    "Modern_Family_ep1": 6615,
    "Modern_Family_ep2": 6210,
    "Modern_Family_ep3": 6110,
    "Modern_Family_ep4": 6235,
    "Modern_Family_ep5": 6040,
    "Modern_Family_ep6": 5990,
    "Sons_of_Anarchy_ep1": 16455,
    "Sons_of_Anarchy_ep2": 13350,
    "Sons_of_Anarchy_ep3": 13685,
}

CLASS2IDX = {
    c: i
    for i, c in enumerate(
        [
            "Answer phone",
            "Clap",
            "Close door",
            "Dress up",
            "Drink",
            "Drive car",
            "Eat",
            "Fall/trip",
            "Fire weapon",
            "Get in/out of car",
            "Give something",
            "Go down stairway",
            "Go up stairway",
            "Hang up phone",
            "Kiss",
            "Open door",
            "Pick something up",
            "Point",
            "Pour",
            "Punch",
            "Read",
            "Run",
            "Sit down",
            "Smoke",
            "Stand up",
            "Throw something",
            "Undress",
            "Use computer",
            "Wave",
            "Write",
        ]
    )
}


def prep_tvseries_anno(orig_anno_dir: str, output_pickle_path: str, fps: float = 5.0):
    orig_anno_dir = Path(orig_anno_dir)
    assert orig_anno_dir.exists()

    sec2idx = 25 / fps  # original sampling rate was 25 fps

    # Predefine with zeros acconding to length
    anno = {
        k: np.zeros((length, NUM_CLASSES), dtype=np.bool)
        for k, length in VIDEO_LENGTHS.items()
    }

    for anno_file_name in ["GT-train.txt", "GT-val.txt", "GT-test.txt"]:
        anno_file = orig_anno_dir / anno_file_name
        assert anno_file.exists()

        # Columns:
        # 0 series name and episode number (Name_Of_The_Series_epNUMBER)
        # 1 class name (same as in classes.txt)
        # 2 start of the action (in seconds)
        # 4 end of the action (in seconds)
        # 5 only one person visible? (yes/no (1/0))
        # 6 atypical action instance (the person does something that is unusual for this action)? (yes/no)
        # 7 shotcut during action? (yes/no)
        # 8 moving camera during action? (yes/no)
        # 9 person is very small or in background? (yes/no)
        # 10 action is recorded from the front? (yes/no)
        # 11 action is recorded from the side? (yes/no)
        # 12 action is recorded from an unusual viewpoint? (yes/no)
        # 13 action is (partly) spatially occluded (by object/person/...)? (yes/no)
        # 14 action is spatially truncated (by frame border)? (yes/no)
        # 15 beginning of the action is missing? (yes/no)
        # 16 end of the action is missing? (yes/no)
        # 17 comments concerning the content of the action or the degree of occlusion and truncation (optional)
        print(f"Loading annotation file {anno_file_name}")
        df = pd.read_table(str(anno_file))

        # Output format: multi-hot encoding for each time-step

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="parsing"):
            class_index = CLASS2IDX[row[1]]
            start_idx = round(row[2] * sec2idx)
            end_idx = round(row[3] * sec2idx)
            anno[row[0]][start_idx:end_idx, class_index] = 1

    # For each time-step without any class, mark class_index 0
    print("Ensuring non-zero step annotataion")
    for a in anno.values():
        a[a.sum(1) == 0, 0] = 1

    # Save to file
    print(f"Saving annotation to {output_pickle_path}")
    with open(output_pickle_path, "wb") as f:
        pickle.dump(anno, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Job's done 🎉")


if __name__ == "__main__":
    parser = ArgumentParser(
        """TVSeries annotation parser.
        The original annotations are converted to dense annotations with a regular time-step.
        Class with index 0 represents that no other class is present.
        The other classes are sorted in alphabetic order."""
    )
    parser.add_argument(
        "--orig_anno_dir",
        type=str,
        help="Directory for original TVSeries annotation",
    )
    parser.add_argument(
        "--output_pickle_path",
        type=str,
        help="Path to output pickle file with parsed annotations",
    )
    parser.add_argument(
        "--fps",
        default=5.0,
        type=float,
        help="Annotation frames per second",
    )
    args = parser.parse_args()

    prep_tvseries_anno(
        args.orig_anno_dir,
        args.output_pickle_path,
        fps=args.fps,
    )
