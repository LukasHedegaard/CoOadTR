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
        if "co3d" in self.feature_pretrain:
            feature_cox3d = pickle.load(
                open(
                    osp.join(
                        self.pickle_root,
                        "co3d/cox3d_s_tvseries.pickle",
                    ),
                    "rb",
                )
            )
            feature_cox3d = {
                k: v.squeeze(0).transpose(0, 1) for k, v in feature_cox3d.items()
            }
            feature_effnet = pickle.load(
                open(
                    osp.join(
                        self.pickle_root,
                        "co3d/efficientnet_b5_tvseries.pickle",
                    ),
                    "rb",
                )
            )

            sample_name = next(iter(feature_cox3d))
            transient_frames = (
                feature_effnet[sample_name].shape[0]
                - feature_cox3d[sample_name].shape[0]
            )

            self.feature_All = {
                k: {
                    "rgb": feature_effnet[k][transient_frames:],
                    "flow": feature_cox3d[k],
                }
                for k in feature_cox3d
            }

            print(
                f"loaded co3d features with dim_size {feature_effnet[sample_name].shape[1]} + {feature_cox3d[sample_name].shape[1]} = {feature_effnet[sample_name].shape[1] + feature_cox3d[sample_name].shape[1]}"
            )

            # Ensure the same number of frames in annotation
            target_all = {
                k: target_all[k][-(len(v["rgb"])) :]
                for k, v in self.feature_All.items()
            }

            self.sessions = list(target_all.keys())
        else:
            assert (Path(self.pickle_root) / self.feature_pretrain).suffix in {
                ".pickle",
                ".pkl",
            }
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
            # target = np.load(osp.join(self.data_root, 'target', session+'.npy'))  # thumos_val_anno.pickle
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
        if "V3" in self.feature_pretrain:
            if osp.exists(
                osp.join(
                    self.pickle_root,
                    "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                )
            ):
                self.feature_All = pickle.load(
                    open(
                        osp.join(
                            self.pickle_root,
                            "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                        ),
                        "rb",
                    )
                )
                print("load thumos_all_feature_{}_V3.pickle !".format(self.subnet))
            else:
                self.feature_All = {}
                for session in self.sessions:
                    self.feature_All[session] = {}
                    self.feature_All[session]["rgb"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_rgb.npy"
                        )
                    )
                    self.feature_All[session]["flow"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_flow.npy"
                        )
                    )
                with open(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.feature_All, f)
                print("dump thumos_all_feature_{}_V3.pickle !".format(self.subnet))
        elif "Anet2016_feature_v2" in self.feature_pretrain:
            if self.resize:
                if osp.exists(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}_tsn_v2_resize.pickle".format(
                            self.subnet
                        ),
                    )
                ):
                    self.feature_All = pickle.load(
                        open(
                            osp.join(
                                self.pickle_root,
                                "thumos_all_feature_{}_tsn_v2_resize.pickle".format(
                                    self.subnet
                                ),
                            ),
                            "rb",
                        )
                    )
                    print(
                        "load thumos_all_feature_{}_tsn_v2_resize.pickle !".format(
                            self.subnet
                        )
                    )
            else:
                if osp.exists(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}_tsn_v2.pickle".format(self.subnet),
                    )
                ):
                    self.feature_All = pickle.load(
                        open(
                            osp.join(
                                self.pickle_root,
                                "thumos_all_feature_{}_tsn_v2.pickle".format(
                                    self.subnet
                                ),
                            ),
                            "rb",
                        )
                    )
                    print(
                        "load thumos_all_feature_{}_tsn_v2.pickle !".format(self.subnet)
                    )
                else:
                    self.feature_All = {}
                    for session in self.sessions:
                        self.feature_All[session] = {}
                        self.feature_All[session]["rgb"] = np.load(
                            osp.join(
                                self.data_root,
                                self.feature_pretrain,
                                session + "_rgb.npy",
                            )
                        )
                        self.feature_All[session]["flow"] = np.load(
                            osp.join(
                                self.data_root,
                                self.feature_pretrain,
                                session + "_flow.npy",
                            )
                        )
                    with open(
                        osp.join(
                            self.pickle_root,
                            "thumos_all_feature_{}_tsn_v2.pickle".format(self.subnet),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(self.feature_All, f)
                    print(
                        "dump thumos_all_feature_{}_tsn_v2.pickle !".format(self.subnet)
                    )
        else:
            if osp.exists(
                osp.join(
                    self.pickle_root, "thumos_all_feature_{}.pickle".format(self.subnet)
                )
            ):
                self.feature_All = pickle.load(
                    open(
                        osp.join(
                            self.pickle_root,
                            "thumos_all_feature_{}.pickle".format(self.subnet),
                        ),
                        "rb",
                    )
                )
                print("load thumos_all_feature_{}.pickle !".format(self.subnet))
            else:
                self.feature_All = {}
                for session in self.sessions:
                    self.feature_All[session] = {}
                    self.feature_All[session]["rgb"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_rgb.npy"
                        )
                    )
                    self.feature_All[session]["flow"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_flow.npy"
                        )
                    )
                with open(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}.pickle".format(self.subnet),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.feature_All, f)
                print("dump thumos_all_feature_{}.pickle !".format(self.subnet))

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
