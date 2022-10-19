# -*- coding: UTF-8 -*-
# 全使用numpy的random策略
import io
import gzip
import base64
import sys
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, get_dimensions
from .base import Txt2ImgIterableBaseDataset

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
            caption, _, img_b64 = vec[:3]
        elif data_source == "yfcc":
            caption, _, _, _, img_b64 = vec[:5]
        elif data_source == "img1024_cc15m_laion400m":
            caption, _, _, _, img_b64 = vec[:5]
        elif data_source == 'vc':
            _, _, _, img_b64, _, _, caption = vec[:7]
        else:
            _, captions, _ , _, _, img_b64 = vec[:6]
            caption = np.random.sample(captions.split("|"), 1)[0].replace("\1", "")

        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
        if np.random.random() < 0.1:
            caption = ''
        return dict(
            image=image,
            caption=caption
        )
    except:
        print(f'error when parse file {filename}')
        return None

class RandomCrop(transforms.RandomCrop):
    @staticmethod
    def get_params(img, output_size):
        _, h, w = get_dimensions(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w
        # 使用这个为了与paddle保持一致。
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

class TextImagePair(Txt2ImgIterableBaseDataset):
    def __init__(self,
                file_list,
                size,
                num_records,
                image_processing=None,
                buffer_size=1000,
                shuffle_every_n_samples=5
        ):
        if image_processing is None:
            self.image_processing = transforms.Compose([
                transforms.Resize(int(size/0.9), InterpolationMode.LANCZOS),
                RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(.5, .5),
            ])
        else:
            self.image_processing = image_processing
        self.file_list = []
        with open(file_list, 'r') as f:
            file_list = f.read().strip().split('\n')
            for file_l in file_list:
                with open(file_l, 'r') as f:
                    self.file_list.append(f.read().strip().split('\n'))
        print([len(file_l) for file_l in self.file_list])

        self.num_records = num_records
        self.file_ids = [np.arange(len(filelist)) for filelist in self.file_list]
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples

    def sample_loader(self, file_ids, filenames):
        while True:
            np.random.shuffle(file_ids)
            for i in file_ids:
                filename = filenames[i].strip("\n")
                with gzip.open(filename, 'rb') if filename.endswith('.gz') else open(filename, 'rb') as f:
                    retry = 0
                    while True:
                        line = f.readline()

                        if line == b'':
                            break
                        try:
                            line = line.decode(encoding='utf-8')
                        except:
                            line = line.decode(encoding='gb18030')
                        data = parse_line(line, filename)
                        if data is None:
                            retry += 1
                            if retry > 100:
                                break
                            continue
                        else:
                            data['image'] = self.image_processing(data['image'])
                            yield data

    def random_load_from_multi_dataset(self):
        print([len(f) for f in self.file_ids])
        sample_loader_per_dataset = [
            iter(self.sample_loader(self.file_ids[i], self.file_list[i]))
            for i in range(len(self.file_ids))
        ]

        while True:
            sample_loader = np.random.choice(sample_loader_per_dataset)
            yield next(sample_loader)

    def shuffle(self, iterator):
        buffer_list = []
        for _ in range(self.buffer_size):
            buffer_list.append(next(iterator))
        i = 0
        while True:
            if i % self.shuffle_every_n_samples == 0:
                np.random.shuffle(buffer_list)
            yield buffer_list.pop()
            buffer_list.append(next(iterator))
            i += 1

    def __len__(self):
        return self.num_records

    def __iter__(self):
        return self.shuffle(iter(self.random_load_from_multi_dataset()))


def worker_init_fn(_):  # for debug, real worker_init_fn in main.py
    worker_info = torch.utils.data.get_worker_info()
    local_rank = 0
    world_size = 1
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    worker_global_id = local_rank * num_workers + worker_id

    # dataset.rng = np.random.RandomState(worker_global_id)
    for i in range(len(dataset.file_ids)):

        file_ids = dataset.file_ids[i]
        num_chunks = world_size * num_workers
        chunk_size = len(file_ids) // num_chunks

        begin_id = worker_global_id * chunk_size
        end_id = (worker_global_id + 1) * chunk_size
        dataset.file_ids[i] = dataset.file_ids[i][begin_id: end_id]
        print(f'dataset {i}, local_rank: {local_rank}, worker_id: {worker_id}, worker_global_id: {worker_global_id}, file_range: ({begin_id}, {end_id})')




if __name__ == '__main__':
    import torch
    import importlib
    importlib.reload(sys)
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    from tqdm import trange

    dataset = TextImagePair(
        'data/filelist/laion400m.visual_chinese.filelist.list',
        256, 1000, buffer_size=100)
    data_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=6,
        worker_init_fn=worker_init_fn,
    )

    data_iter = iter(data_loader)
    for i in trange(50):
        data = next(data_iter)
    print(data['image'].size())
    print(data['caption'])



