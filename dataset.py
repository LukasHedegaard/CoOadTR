from os import minor
import os.path as osp
import pickle5 as pickle
import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path


class TRNTVSeriesDataLayer(data.Dataset):
    def __init__(self, args, phase="train"):
        self.pickle_root = "data/"
        self.sessions = getattr(args, phase + "_session_set")  # video name
        self.enc_steps = args.enc_layers
        self.numclass = args.numclass
        self.dec_steps = args.query_num
        self.training = phase == "train"
        self.feature_pretrain = args.feature
        assert (Path(self.pickle_root) / self.feature_pretrain).suffix in {
            ".pickle",
            ".pkl",
        }
        self.inputs = []

        # Load anno
        target_all = pickle.load(
            open(
                osp.join(self.pickle_root, "tvseries_anno.pickle"),
                "rb",
            )
        )

        # Load features
        assert osp.exists(osp.join(self.pickle_root, self.feature_pretrain))
        self.feature_All = pickle.load(
            open(
                osp.join(self.pickle_root, self.feature_pretrain),
                "rb",
            )
        )

        # Ensure that features and annotataions have same length
        for k, v in self.feature_All.items():
            min_len = min(v["rgb"].shape[0], v["flow"].shape[0])
            if v["rgb"].shape[0] != min_len:
                v["rgb"] = v["rgb"][:min_len]

            if v["flow"].shape[0] != min_len:
                v["flow"] = v["flow"][:min_len]

            if target_all[k].shape[0] != min_len:
                target_all[k] = target_all[k][:min_len]

        for session in self.sessions:
            target = target_all[session]

            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0], 1),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, 1),
            ):
                enc_target = target[start:end]
                dec_target = target[end : end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(
                    target[start:end]
                )
                self.inputs.append(
                    [
                        session,
                        start,
                        end,
                        enc_target,
                        distance_target,
                        class_h_target,
                        dec_target,
                    ]
                )

        print(f"Loaded {self.feature_pretrain}")

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros(
            (self.enc_steps, self.dec_steps, target_vector.shape[-1])
        )
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        target_argmax = target_vector[self.enc_steps - 1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.0
        return target_matrix, target_vector[self.enc_steps - 1]

    def __getitem__(self, index):
        """self.inputs.append([
            session, start, end, enc_target, distance_target, class_h_target
        ])"""
        (
            session,
            start,
            end,
            enc_target,
            distance_target,
            class_h_target,
            dec_target,
        ) = self.inputs[index]
        camera_inputs = self.feature_All[session]["rgb"][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        motion_inputs = self.feature_All[session]["flow"][start:end]
        motion_inputs = torch.tensor(motion_inputs)
        enc_target = torch.tensor(enc_target)
        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)
        return (
            camera_inputs,
            motion_inputs,
            enc_target,
            distance_target,
            class_h_target,
            dec_target,
        )

    def __len__(self):
        return len(self.inputs)


class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase="train"):
        args.data_root = "~/datasets/thumos14/features"
        self.pickle_root = "data/"
        self.data_root = args.data_root  # data/THUMOS
        self.sessions = getattr(args, phase + "_session_set")  # video name
        self.enc_steps = args.enc_layers
        self.numclass = args.numclass
        self.dec_steps = args.query_num
        self.training = phase == "train"
        self.feature_pretrain = (
            args.feature
        )  # 'Anet2016_feature'   # IncepV3_feature  Anet2016_feature
        self.inputs = []

        self.subnet = "val" if self.training else "test"
        self.with_audio = "audio" in self.feature_pretrain
        self.resize = args.resize_feature
        if self.resize:
            target_all = pickle.load(
                open(
                    osp.join(
                        self.pickle_root,
                        "thumos_" + self.subnet + "_anno_resize.pickle",
                    ),
                    "rb",
                )
            )
        else:
            target_all = pickle.load(
                open(
                    osp.join(
                        self.pickle_root, "thumos_" + self.subnet + "_anno.pickle"
                    ),
                    "rb",
                )
            )
        for session in self.sessions:  # 改
            target = target_all[session]["anno"]
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0], 1),  # self.enc_steps
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, 1),
            ):
                enc_target = target[start:end]
                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end : end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(
                    target[start:end]
                )
                if class_h_target.argmax() != 21:
                    self.inputs.append(
                        [
                            session,
                            start,
                            end,
                            enc_target,
                            distance_target,
                            class_h_target,
                            dec_target,
                        ]
                    )

        file_prefix = "thumos_"
        if "kin" in self.feature_pretrain:
            file_prefix += "kin_"
        else:
            assert "anet" in self.feature_pretrain
            file_prefix += "anet_"

        if self.with_audio:
            file_prefix += "plus_audio_"

        file_path = osp.join(self.pickle_root, f"{file_prefix}{self.subnet}.pickle")
        assert osp.exists(file_path)
        self.feature_All = pickle.load(open(file_path, "rb"))
        print(f"Loaded {file_path}")

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros(
            (self.enc_steps, self.dec_steps, target_vector.shape[-1])
        )
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        target_argmax = target_vector[self.enc_steps - 1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.0
        return target_matrix, target_vector[self.enc_steps - 1]

    def __getitem__(self, index):
        """self.inputs.append([
            session, start, end, enc_target, distance_target, class_h_target
        ])"""
        (
            session,
            start,
            end,
            enc_target,
            distance_target,
            class_h_target,
            dec_target,
        ) = self.inputs[index]
        camera_inputs = self.feature_All[session]["rgb"][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        motion_inputs = self.feature_All[session]["flow"][start:end]
        motion_inputs = torch.tensor(motion_inputs)

        # A bit dirty: To avoid downstream modifications, we'll merge audio into flow
        if self.with_audio:
            audio_inputs = self.feature_All[session]["audio"][start:end]
            audio_inputs = torch.tensor(audio_inputs)
            # motion_inputs = torch.concat((motion_inputs, audio_inputs), dim=1)
            motion_inputs = audio_inputs

        enc_target = torch.tensor(enc_target)
        distance_target = torch.tensor(distance_target)
        class_h_target = torch.tensor(class_h_target)
        dec_target = torch.tensor(dec_target)
        return (
            camera_inputs,
            motion_inputs,
            enc_target,
            distance_target,
            class_h_target,
            dec_target,
        )

    def __len__(self):
        return len(self.inputs)
