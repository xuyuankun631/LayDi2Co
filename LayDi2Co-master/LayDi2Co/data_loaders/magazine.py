import json
from functools import partial

import numpy as np
from torch.utils.data.dataset import Dataset

from data_loaders.data_utils import is_valid_comp, norm_bbox, mask_whole_box, mask_loc, mask_size, mask_cat, \
    mask_random_box_and_cat, mask_all
from utils import getDistinctColors


class MagazineLayout(Dataset):
    component_class = {'text': 0, 'image': 1, 'headline': 2, 'text-over-image': 3,
                       'headline-over-image': 4}
    idx2class = {v + 1: k for k, v in component_class.items()}

    colors_f = getDistinctColors(len(component_class) + 2)
    name2color_map = {}
    idx2color_map = {}
    for i, c in enumerate(colors_f):
        if i == 0:
            name2color_map["empty_token"] = c
            idx2color_map[0] = c
        elif i == len(component_class) + 1:
            name2color_map["drop"] = c
            idx2color_map[len(component_class) + 1] = c
        else:
            name2color_map[idx2class[i]] = c
            idx2color_map[i] = c
    # mask 函数映射
    mask_func_map = {
        'whole_box': mask_whole_box,
        'loc': mask_loc,
        'size': mask_size,
        'cat': mask_cat,
        'random_box_and_cat': mask_random_box_and_cat,
        'all': mask_all
    }




    #  json_path（JSON 文件路径）、max_num_com（最大组件数量）和 cond_type（条件类型）作为参数。
    def __init__(self, json_path: str, max_num_com: int = 9, cond_type=None):
        self.categories_num = len(self.component_class.keys()) + 2
        self.data = {"bbox": [],
                     "file_idx": [],
                     "annotations": []}
        self.max_num_comp = max_num_com
        self.json_path = json_path
        self.process()
        self.cond_type = cond_type

    # 加载并处理来自提供的 JSON 文件的数据
    # 构建图像 ID 与其注释和类别的映射
    # 根据图像尺寸筛选和标准化边界框
    # 确保每张图片的组件数量在设定的最大值 max_num_com 内。
    # 对边界框和类别进行标准化和随机打乱，并进行缩放处理，以便用于模型训练
    def process(self):
        json_data = json.load(open(self.json_path, 'r'))
        img2ann = {}
        for t in json_data['annotations']:
            if not t['image_id'] in img2ann.keys():
                img2ann[t['image_id']] = []
            img2ann[t['image_id']].append(t)
        ind2cat = {}
        # 构建图像 ID 与其注释和类别的映射
        for t in json_data['categories']:
            ind2cat[t['id']] = t['name']

        # 构建图像 ID 与其注释和类别的映射
        img_ids = [t['id'] for t in json_data['images']]
        sort_img_ind = [i[0] for i in sorted(enumerate(img_ids), key=lambda x: x[1])]
        # 根据图像尺寸筛选和标准化边界框
        for id in sort_img_ind:
            ann_img = json_data['images'][id]
            W = float(ann_img['width'])
            H = float(ann_img['height'])

            elements = img2ann[img_ids[id]]
            elements = list(filter(partial(is_valid_comp, W=W, H=H), elements))

            # 确保每张图片的组件数量在设定的最大值 max_num_com 内。
            N = len(elements)
            if N <= 1 or N > self.max_num_comp:
                continue

            boxes = []
            categories = []
            for element in elements:
                # normalize bboxes
                b = norm_bbox(H, W, element)
                boxes.append(b)
                category = ind2cat[element["category_id"]]
                categories.append(self.component_class[category])

            ann_box = np.array(boxes)
            ann_cat = np.array(categories)
            # we want to get order invariant model
            ind = [ttt for ttt in range(ann_box.shape[0])]
            # 对边界框和类别进行标准化和随机打乱，并进行缩放处理，以便用于模型训练
            np.random.shuffle(ind)
            ann_cat = ann_cat[ind]
            # shift class by 1 because 0 is for empty
            ann_cat += 1
            ann_box = ann_box[ind]
            # scale
            ann_box = ((ann_box * 2) - 1) * 2
            self.data["bbox"].append(ann_box)
            self.data["annotations"].append(ann_cat)
            self.data["file_idx"].append(ann_img['file_name'])

    # 根据给定的索引返回边界框、类别、索引和文件名，同时随机打乱边界框顺序，以增加数据的随机性。
    def get_data_by_ix(self, idx):
        box = self.data['bbox'][idx]
        ind = list(range(box.shape[0]))
        np.random.shuffle(ind)
        box = box[ind]
        cat = self.data['annotations'][idx][ind]

        name = self.data['file_idx'][idx]
        return box, cat, ind, name

    # 根据索引获取并处理数据。
    def process_data(self, idx):
        box, cat, ind, name = self.get_data_by_ix(idx)

        # 基于边界框进行遮罩，并将数据进行填充，确保一致的输入尺寸
        mask, mask4cat = self.mask_instance(box)

        box, cat, mask, mask4cat = self.pad_instance(box, cat, mask, mask4cat, self.max_num_comp)
        # 返回一个包含处理后数据的字典（包括 box、cat、mask_box、mask_cat）
        return {
            "box": box.astype(np.float32),
            "cat": cat.astype(int),
            "box_cond": box.copy().astype(np.float32),
            "mask_box": mask.astype(int),
            "mask_cat": mask4cat.astype(int),
        }

    # 静态方法 mask_instance：
    # 定义了一种遮罩策略，可以应用于边界框
    # 从预定义的遮罩函数中随机选择一个，并根据选择的函数调整遮罩（如位置、大小、类别等）。
    @staticmethod
    def mask_instance(box):
        # here you can implement any masking strategy
        # mask, mask4cat = mask_all(box.shape)
        # mask_func = np.random.choice([mask_loc, mask_whole_box, mask_all], size=1,
        #                              p=[0.35, 0.35, 0.3])[0]
        # if mask_func == mask_all:
        #     mask, mask4cat = mask_func(box.shape)
        # else:
        #     r_mask = 1.0
        #     mask, mask4cat = mask_func(box.shape, r_mask=r_mask)
        mask_func = np.random.choice([mask_loc, mask_size, mask_cat, mask_whole_box, mask_random_box_and_cat,
                                      mask_all], 1,
                                     p=[0.2, 0.1, 0.05, 0.25, 0.1, 0.3])[0]
        if mask_func == mask_random_box_and_cat:
            r_mask_box = np.random.uniform(0.5, 1.0, size=1)[0]
            r_mask_cat = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask_box=r_mask_box, r_mask_cat=r_mask_cat)
        elif mask_func == mask_all:
            mask, mask4cat = mask_func(box.shape)
        else:
            r_mask = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask=r_mask)
        return mask, mask4cat

    # 将边界框、类别和遮罩填充到一个固定大小（max_num_comp），确保输入尺寸的一致性，避免数据维度不匹配。
    @staticmethod
    def pad_instance(box, cat, mask, mask4cat, max_num_comp=9):
        box = np.pad(box, pad_width=((0, max_num_comp - box.shape[0]), (0, 0)), constant_values=0.)
        cat = np.pad(cat, pad_width=(0, max_num_comp - cat.shape[0]), constant_values=0.)
        mask = np.pad(mask, pad_width=((0, max_num_comp - mask.shape[0]), (0, 0)), constant_values=0.)
        mask4cat = np.pad(mask4cat, pad_width=(0, max_num_comp - mask4cat.shape[0]),
                          constant_values=0.)
        return box, cat, mask, mask4cat

    # 根据索引获取处理后的数据。
    # 如果指定了 cond_type，则应用条件处理；否则调用常规的处理方法
    def __getitem__(self, idx):
        if self.cond_type:
            assert self.cond_type in self.mask_func_map.keys()
            return self.process_data_cond(idx, self.cond_type)
        sample = self.process_data(idx)
        return sample

    # 返回数据集的大小（即图像/边界框条目的数量）
    def __len__(self):
        return len(self.data['bbox'])

    # 类似于 process_data，但根据 cond_type 应用不同的遮罩策略（例如可以根据条件类型应用不同的遮罩函数）
    def process_data_cond(self, idx, cond_type):
        box, cat, ind, name = self.get_data_by_ix(idx)

        mask_func = self.mask_func_map[cond_type]
        if mask_func == mask_random_box_and_cat:
            r_mask_box = np.random.uniform(0.5, 1.0, size=1)[0]
            r_mask_cat = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask_box=r_mask_box, r_mask_cat=r_mask_cat)
        elif mask_func == mask_all:
            mask, mask4cat = mask_func(box.shape)
        else:
            mask, mask4cat = mask_func(box.shape, r_mask=1.0)
        box, cat, mask, mask4cat = self.pad_instance(box, cat, mask, mask4cat, self.max_num_comp)
        return {
            "box": box.astype(np.float32),
            "cat": cat.astype(int),
            "box_cond": box.copy().astype(np.float32),
            "mask_box": mask.astype(int),
            "mask_cat": mask4cat.astype(int),
        }