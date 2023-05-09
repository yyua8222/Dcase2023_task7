# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import sys

sys.path.append(
    "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio"
)

import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import utilities.audio as Audio
import librosa
import os
import torchvision
import yaml
import pandas as pd

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


class Dataset(Dataset):
    def __init__(
        self,
        preprocess_config,
        train_config,
        samples_weight=None,
        train=True,
        shuffle=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        self.datapath = (
            preprocess_config["path"]["train_data"]
            if (train)
            else preprocess_config["path"]["test_data"]
        )
        self.transform = None

        self.data = []
        if type(self.datapath) is str:
            # with open(self.datapath, "r") as fp:    # change for our own json style
            self.data = [json.loads(line) for line in open(self.datapath, 'r')]

            #     data_json = json.load(fp)
            # self.data = data_json["data"]
        elif type(self.datapath) is list:
            for datapath in self.datapath:
                with open(datapath, "r") as fp:
                    data_json = json.load(fp)
                self.data += data_json["data"]
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.samples_weight = samples_weight

        # if(self.samples_weight is not None):
        #     print("+Use balance sampling on the mixup audio")
        #     self.sample_weight_index = list(range(len(self.samples_weight)))
        #     self.samples_weight /= np.sum(self.samples_weight)

        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]

        # try:
        #     self.rolling = train_config["augmentation"]["rolling"]
        #     if(self.rolling):
        #         print("+ Using rolling augmentation")
        # except:
        #     self.rolling = False

        # if(self.mixup > 0 and self.samples_weight is not None):
        #     self.mix_sample_idx_queue = np.random.choice(self.sample_weight_index, p=self.samples_weight, size=1000)

        # No augmentation during evaluation
        if train == False:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.sampling_rate = preprocess_config["preprocessing"]["audio"][
            "sampling_rate"
        ]
        # self.segment_label_path = preprocess_config["path"]["segment_label_path"]
        # self.clip_label_path = preprocess_config["path"]["clip_label_path"]
        self.hopsize = self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.target_length = self.preprocess_config["preprocessing"]["mel"][
            "target_length"
        ]
        self.use_blur = self.preprocess_config["preprocessing"]["mel"]["blur"]

        # try: self.label_norm = self.preprocess_config["preprocessing"]["label"]["norm"]
        # except: self.label_norm=False
        # try: self.label_threshold = self.preprocess_config["preprocessing"]["label"]["threshold"]
        # except: self.label_threshold=False

        self.segment_length = int(self.target_length * self.hopsize)

        try:
            self.segment_size = self.preprocess_config["preprocessing"]["audio"][
                "segment_size"
            ]
            self.target_length = int(self.segment_size / self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None

        # try:
        #     self.label_use_original_ground_truth = self.preprocess_config["preprocessing"]["label"]["label_use_original_ground_truth"]
        #     if(self.label_use_original_ground_truth): print("==> Use ground truth label: %s" % self.label_use_original_ground_truth)
        # except:
        #     print("Use machine labels")
        #     self.label_use_original_ground_truth=False

        # try:
        #     self.label_use_both_original_gt_and_machine_labels = self.preprocess_config["preprocessing"]["label"]["label_use_both_original_gt_and_machine_labels"]
        #     if(self.label_use_both_original_gt_and_machine_labels): print("==> Use both ground truth label and machine labels at the same time: %s" % self.label_use_both_original_gt_and_machine_labels)
        # except:
        #     self.label_use_both_original_gt_and_machine_labels=False

        print(
            "Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s"
            % (self.mixup, self.timem, self.freqm, self.use_blur)
        )

        # dataset spectrogram mean and std, used to normalize the input
        # self.norm_mean = preprocess_config["preprocessing"]["mel"]["mean"]
        # self.norm_std = preprocess_config["preprocessing"]["mel"]["std"]

        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = False
        self.noise = False
        if self.noise == True:
            print("now use noise augmentation")

        self.index_dict = make_index_dict(
            preprocess_config["path"]["class_label_index"]
        )
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

        # self.class_reweight_matrix = np.load(preprocess_config["path"]["class_reweight_arr_path"])

        self.id2label, self.id2num, self.num2label = self.build_id_to_label()

    def build_id_to_label(self):
        ret = {}
        id2num = {}
        num2label = {}
        df = pd.read_csv(self.preprocess_config["path"]["class_label_index"])
        for _, row in df.iterrows():
            index, mid, display_name = row["index"], row["mid"], row["display_name"]
            ret[mid] = display_name
            id2num[mid] = index
            num2label[index] = display_name
        return ret, id2num, num2label

    def resample(self, waveform, sr):
        if sr == 22050:
            if self.sampling_rate == 22050:
                return waveform
            elif self.sampling_rate == 16000:
                if self.transform:
                    waveform = self.transform(waveform)

                else:
                    self.transform = torchaudio.transforms.Resample(22050,16000)
                    waveform = self.transform(waveform)

                # waveform = torchaudio.transforms.Resample()
            return waveform
        if sr == 32000 and self.sampling_rate == 16000:
            waveform = waveform[::2]
            return waveform
        if sr == 48000 and self.sampling_rate == 16000:
            waveform = waveform[::3]
            return waveform
        else:
            raise ValueError(
                "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
                % (sr, self.sampling_rate)
            )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5

    def random_segment_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - self.segment_length) <= 0:
            return waveform

        random_start = int(
            self.random_uniform(0, waveform_length - self.segment_length)
        )
        return waveform[:, random_start : random_start + self.segment_length]

    def pad_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == self.segment_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, self.segment_length))
        rand_start = int(self.random_uniform(0, self.segment_length - waveform_length))
        # rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower



        waveform, sr = torchaudio.load(filename)  # Faster!!!
        # print(f"the file name is {filename} and the sr is {sr} and the samplying rate is {self.sampling_rate}")

        # print(f"the current waveform is {waveform.shape} and sr is {sr} and sampleing rate is {self.sampling_rate}")
        waveform = self.resample(waveform, sr)
        # print(f"the new waveform shape is {waveform.shape}")
        waveform = waveform.numpy()[0, ...]
        waveform = self.normalize_wav(waveform)
        waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]

        waveform = self.random_segment_wav(waveform)
        waveform = self.pad_wav(waveform)

        return waveform

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform = self.read_wav_file(filename)
        # mixup
        else:
            waveform1 = self.read_wav_file(filename)
            waveform2 = self.read_wav_file(filename2)

            mix_lambda = np.random.beta(5, 5)
            mix_waveform = mix_lambda * waveform1 + (1
                                                     - mix_lambda) * waveform2
            waveform = self.normalize_wav(mix_waveform)

        # if self.segment_length > waveform.shape[1]:
        #     # padding
        #     temp_wav = np.zeros((1, self.segment_length))
        #     temp_wav[:, :waveform.shape[1]] = waveform
        #     waveform = temp_wav
        # else:
        #     # cutting
        #     waveform = waveform[:, :self.segment_length]

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        fbank, log_magnitudes_stft, energy = Audio.tools.get_mel_from_wav(
            waveform, self.STFT
        )

        fbank = torch.FloatTensor(fbank.T)
        log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

        fbank, log_magnitudes_stft = self._pad_spec(fbank), self._pad_spec(
            log_magnitudes_stft
        )

        if filename2 == None:
            return fbank, log_magnitudes_stft, 0, waveform
        else:
            return fbank, log_magnitudes_stft, mix_lambda, waveform

    def _pad_spec(self, fbank):
        n_frames = fbank.shape[0]
        #
        # print(f"the n_frames is {n_frames}")
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        if fbank.size(-1) % 2 != 0:
            fbank = fbank[..., :-1]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # if(self.label_use_both_original_gt_and_machine_labels):
        #     if(self.make_decision(0.5)):
        #         self.label_use_original_ground_truth = True
        #     else:
        #         self.label_use_original_ground_truth = False

        (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        ) = self.feature_extraction(index)

        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += self.label_indices_to_text(mix_datum, label_indices)

        t_step = fbank.size(0)
        waveform = waveform[..., : int(self.hopsize * t_step)]

        # if(self.rolling and self.make_decision(1.0)):
        #     rand_roll = int(self.random_uniform(0, t_step))
        #     fbank = torch.roll(fbank, shifts=rand_roll, dims=0)
        #     log_magnitudes_stft = torch.roll(log_magnitudes_stft, shifts=rand_roll, dims=0)
        #     waveform = torch.roll(waveform, shifts = rand_roll * self.hopsize, dims=-1)

        # fbank = self.aug(fbank)

        # Reconsider whether or not need this step?
        # if(not self.label_use_original_ground_truth):
        # seg_label = self.process_labels(seg_label)
        # else:

        # if(self.label_use_original_ground_truth):
        #     if(len(label_indices.shape) <= 1):
        #         seg_label = label_indices[None,...]
        #     seg_label = np.repeat(seg_label.numpy(), 1056, 0)
        #     seg_label = seg_label[:self.target_length,:]
        #     clip_label = label_indices

        return (
            fbank.float(), # mel
            log_magnitudes_stft.float(), # stft
            label_indices.float(), # label
            fname, # fname
            waveform.float(), # waveform
            text, # 
        )  # clip_label.float()

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def label_indices_to_text(self, datum, label_indices):
        if "caption" in datum.keys():
            return datum["caption"]
        name_indices = torch.where(label_indices > 0.1)[0]
        # description_header = "This audio contains the sound of "
        description_header = ""
        labels = ""
        for id, each in enumerate(name_indices):
            if id == len(name_indices) - 1:
                labels += "%s." % self.num2label[int(each)]
            else:
                labels += "%s, " % self.num2label[int(each)]
        return description_header + labels


    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                if random.random() < self.mixup:
                    datum = self.data[index]
                    ###########################################################
                    # if(self.samples_weight is None):
                    mix_sample_idx = random.randint(0, len(self.data) - 1)
                    # else:
                    #     if(self.mix_sample_idx_queue.shape[0] < 10):
                    #         self.mix_sample_idx_queue = np.random.choice(self.sample_weight_index, p=self.samples_weight, size=1000)
                    #     mix_sample_idx = self.mix_sample_idx_queue[-1]
                    #     self.mix_sample_idx_queue = self.mix_sample_idx_queue[:-1]
                    mix_datum = self.data[mix_sample_idx]
                    ###########################################################
                    # get the mixed fbank
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"], mix_datum["wav"]
                    )
                    # initialize the label
                    label_indices = np.zeros(self.label_num)
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += mix_lambda
                    for label_str in mix_datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += (
                            1.0 - mix_lambda
                        )

                else:
                    datum = self.data[index]
                    label_indices = np.zeros(self.label_num)
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"]
                    )
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                    mix_datum = None
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                wav = datum["wav"]
                print(f"feature_extraction error wavfile {wav} and exption {e}")
                continue

        # The filename of the wav file
        fname = datum["wav"]

        # seg_label = torch.FloatTensor(seg_label)
        # clip_label = torch.FloatTensor(clip_label)
        clip_label = None

        return (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        )

    # def read_machine_label(self, index):
    #     # Read the clip-level or segment-level labels
    #     while(True):
    #         try:
    #             clip_label = self.read_label(index)
    #             return clip_label
    #         except Exception as e:
    #             print("read_machine_label", e)
    #             if(index == len(self.data)-1): index = 0
    #             else: index += 1

    def aug(self, fbank):
        assert torch.min(fbank) < 0
        fbank = fbank.exp()
        ############################### Blur and Spec Aug ####################################################
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        # self.use_blur = False
        if self.use_blur:
            fbank = self.blur(fbank)
        if self.freqm != 0:
            fbank = self.frequency_masking(fbank, self.freqm)
        if self.timem != 0:
            fbank = self.time_masking(fbank, self.timem)  # self.timem=0
        #############################################################################################
        fbank = (fbank + 1e-7).log()
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return fbank

    # def read_label(self, index):
    #     if("clip_label" in self.data[index].keys()):
    #         clip_label_fname = self.data[index]["clip_label"]
    #     else:
    #         wav_name = self.data[index]['wav']
    #         label_fname = os.path.basename(wav_name).replace(".wav",".npy")
    #         clip_label_fname = os.path.join(self.clip_label_path, label_fname)

    #     if(not os.path.exists(clip_label_fname)):
    #         return None

    #     clip_label = np.load(clip_label_fname)

    #     # For the clip level label, add one more dimension
    #     if(len(clip_label.shape) <= 1):
    #         clip_label = clip_label[None,...]

    #     clip_label = self.process_labels(clip_label)
    #     # seg_label = self.process_labels(seg_label)

    #     return clip_label

    def __len__(self):
        return len(self.data)

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def blur(self, fbank):
        assert torch.min(fbank) >= 0
        kernel_size = int(self.random_uniform(1, self.melbins))
        fbank = torchvision.transforms.functional.gaussian_blur(
            fbank, kernel_size=[kernel_size, kernel_size]
        )
        return fbank

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        fbank[:, mask_start : mask_start + mask_len, :] *= 0.0
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        fbank[:, :, mask_start : mask_start + mask_len] *= 0.0
        return fbank


def balance_test():
    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_unbal_train_data.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    # dataset = AudioDataset(samples_weight = None, train=True)
    dataset = AudioDataset(samples_weight=samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save(
        "balanced_with_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )
    # np.save("balanced_with_no_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    ######################################
    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save(
        "balanced_with_no_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )

    ######################################

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=8,
        # sampler=sampler
    )

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save("no_balance.npy", label_indices_total.cpu().detach().numpy() / 2000)


def check_batch(batch):
    import soundfile as sf
    import matplotlib.pyplot as plt

    save_path = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/output/temp"
    os.makedirs(save_path, exist_ok=True)
    fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
    for fb, wv, description in zip(fbank, waveform, text):
        sf.write(
            save_path + "/" + "%s.wav" % description.replace(" ", "_")[:30], wv, 16000
        )
        plt.imshow(np.flipud(fb.cpu().detach().numpy().T), aspect="auto")
        plt.savefig(save_path + "/" + "%s.png" % description.replace(" ", "_")[:30])


if __name__ == "__main__":

    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    preprocess_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/preprocess.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )
    train_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/train.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/datafiles/audiocaps_train_label.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(
        samples_weight=samples_weight,
        train=True,
        train_config=train_config,
        preprocess_config=preprocess_config,
    )

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        check_batch(each)
        break
