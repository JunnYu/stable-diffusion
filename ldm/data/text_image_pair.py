# -*- coding: UTF-8 -*-

import traceback
import io
import gzip
import random
import base64
import sys
import json
from abc import abstractmethod
from copy import deepcopy
import glob

import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_tensor, normalize
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset


def parse_line(line, filename):
    def parse_src(filename):
        if 'alt_aesthetic' in filename:
            return 'alt_aesthetic'
        elif 'laion_aesthetic' in filename:
            return 'laion_aesthetic'
        elif "cc_15m_tag_watermark_1024" in filename or "laion400m_1024" in filename:
            return "img1024_cc15m_laion400m"
        elif "cc_tag_watermark" in filename or "cc_12m_tag_watermark" in filename:
            return "cc"
        elif "alt-text2image_gen_filted_data_shuffle_watermark" in filename:
            return "alt"
        elif "image_search_key_201810" in filename:
            return "click"
        elif "laion400m" in filename:
            return "laion400m"
        elif "yfcc_en_zh" in filename:
            return "yfcc"
        elif "vc" in filename:
            return "vc"
        else:
            raise NotImplementedError(f"Unkown data source, {filename}")
    try:
        vec = line.strip().split("\t")
        data_source = parse_src(filename)
        if data_source == 'alt_aesthetic':
            caption, img_b64 = vec[1], vec[4]
        elif data_source == 'laion_aesthetic':
            caption, img_b64 = vec[10], vec[12]
        elif data_source == "cc":
            caption, _, _, _, img_b64 = vec[:5]
            caption = caption.replace('<mark>', '')
        elif data_source == "alt":
            img_b64, caption = vec[:2]
        elif data_source == "laion400m":
            # _, caption, _, img_b64 = vec[:4]
            caption, _, img_b64 = vec[:3]
        elif data_source == "yfcc":
            caption, _, _, _, img_b64 = vec[:5]
        elif data_source == "img1024_cc15m_laion400m":
            caption, _, _, _, img_b64 = vec[:5]
        elif data_source == 'vc':
            _, _, _, img_b64, _, _, caption = vec[:7]
        else:
            _, captions, _ , _, _, img_b64 = vec[:6]
            caption = random.sample(captions.split("|"), 1)[0].replace("\1", "")

        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
        if random.random() < 0.1:
            caption = ''
        return dict(
            image=image,
            caption=caption
        )
    except:
        print(f'error when parse file {filename}')
        traceback.print_exc()
        return None


class TextImagePair(IterableDataset):
    def __init__(self,
                file_list,
                size,
                num_records,
                image_processing=None,
                buffer_size=1000,
                shuffle_every_n_samples=5
        ):
        self.size = size
        if image_processing is None:
            self.image_processing = transforms.Compose([
                transforms.Resize(int(size/0.9), InterpolationMode.LANCZOS),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(.5, .5),
            ])
        else:
            self.image_processing = image_processing
        self.file_list = []
        file_weights = []
        with open(file_list, 'r') as f:
            file_lists = f.read().strip().split('\n')
            for file_l in file_lists:
                file_l = file_l.split(' ')
                if len(file_l) > 1:
                    file_weight = float(file_l[1])
                    file_weights.append(file_weight)
                file_l = file_l[0]
                with open(file_l, 'r') as f:
                    self.file_list.append(f.read().strip().split('\n'))
        print([len(file_l) for file_l in self.file_list])
        if len(file_weights) == len(self.file_list):
            file_weights = np.array(file_weights)
            file_weight_sum = np.sum(file_weights)
            assert file_weight_sum > 0, 'sum of file weights must > 0'
            file_weights = file_weights / file_weight_sum
            print(f'sample weights of files: {file_weights}')
            self.file_weights_cumsum = np.cumsum(file_weights)
            self.file_weights_cumsum = np.concatenate([[0.0], self.file_weights_cumsum])
        else:
            print('sample each file list with same probabiliy')
            self.file_weights_cumsum = None

        self.num_records = num_records
        self.file_ids = [np.arange(len(filelist)) for filelist in self.file_list]
        print(f'original lengths of self.file_ids: {[len(f) for f in self.file_ids]}')
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples

    def sample_loader(self, file_ids, filenames):
        while True:
            random.shuffle(file_ids)
            for i in file_ids:
                filename = filenames[i].strip("\n")
                with gzip.open(filename, 'rb') if filename.endswith('.gz') else open(filename, 'rb') as f:
                    retry = 0
                    while True:
                        line = f.readline()

                        if line == b'':
                            break
                        try: 
                            try:
                                line = line.decode(encoding='utf-8')
                            except:
                                line = line.decode(encoding='gb18030')
                        except:
                            print(f'error on file {filename}')
                            continue
                        data = parse_line(line, filename)
                        if data is None:
                            retry += 1
                            if retry > 100:
                                break
                            continue
                        else:
                            w, h = data['image'].size
                            if w < self.size or h < self.size:
                                continue
                            data['image'] = self.image_processing(data['image'])
                            yield data

    def random_load_from_multi_dataset(self):
        print(f'lengths of self.file_ids in random_load: {[len(f) for f in self.file_ids]}')
        sample_loader_per_dataset = [
            iter(self.sample_loader(self.file_ids[i], self.file_list[i]))
            for i in range(len(self.file_ids))
        ]

        while True:
            if self.file_weights_cumsum is None:
                sample_loader = random.choice(sample_loader_per_dataset)
            else:
                rand_num = random.random()
                for i in range(len(self.file_list)):
                    if self.file_weights_cumsum[i] <= rand_num < self.file_weights_cumsum[i + 1]:
                        break
                sample_loader = sample_loader_per_dataset[i]
                # debug
                # print(self.file_list[i][0])
            yield next(sample_loader)

    def shuffle(self, iterator):
        buffer_list = []
        for _ in range(self.buffer_size):
            buffer_list.append(next(iterator))
        i = 0
        while True:
            if i % self.shuffle_every_n_samples == 0:
                random.shuffle(buffer_list)
            yield buffer_list.pop()
            buffer_list.append(next(iterator))
            i += 1

    def __len__(self):
        return self.num_records

    def __iter__(self):
        return self.shuffle(iter(self.random_load_from_multi_dataset()))
    
    