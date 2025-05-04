import os
import cv2
import json
import torch
import random
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from xml.etree import ElementTree
import pycocotools.mask as mask_utils
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rle_decode(rle, shape):
    """Decodes COCO RLE encoded mask from comma-separated string."""
    try:
        if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
            rle_obj = rle
        elif isinstance(rle, str):
            counts = [int(x.strip()) for x in rle.split(',') if x.strip()]
            if not counts:
                raise ValueError("Empty RLE counts after parsing")
            rle_obj = {'counts': counts, 'size': shape}
        else:
            raise ValueError(f"Invalid RLE format: {rle}")

        counts = [int(x.strip()) for x in rle.split(',') if x.strip()]
        total_pixels = shape[0] * shape[1]
        sum_counts = sum(counts)
        if sum_counts > total_pixels:
            logger.warning(f"RLE counts sum ({sum_counts}) exceeds total pixels ({total_pixels}) for shape {shape}")
            return np.zeros(shape, dtype=np.uint8)

        decoded = mask_utils.frPyObjects(rle_obj, shape[0], shape[1])
        mask = mask_utils.decode(decoded).reshape(shape).astype(np.uint8)
        if mask.sum() == 0:
            logger.warning(f"Decoded RLE mask is empty for shape {shape}")
        return mask
    except Exception as e:
        logger.error(f"Error decoding RLE: {e}, input RLE: {rle}, shape: {shape}")
        return np.zeros(shape, dtype=np.uint8)


class BaseDataset(Dataset):
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.img_files = self._parse_data()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_data = self.img_files[index]
        img_file = img_data['image_path']
        annotation = img_data['annotation']
        return img_file, annotation

    def _parse_data(self):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Annotation file not found at: {self.xml_path}")

        tree = ElementTree.parse(self.xml_path)
        root = tree.getroot()
        img_folder = os.path.dirname(self.xml_path)

        img_files = []
        for image_element in root.findall('image'):
            image_id = image_element.get('id')
            image_name = image_element.get('name')
            image_path = os.path.join(img_folder, image_name)
            width = int(image_element.get('width'))
            height = int(image_element.get('height'))
            annotation = {'width': width, 'height': height, 'labels': {}}
            for mask_element in image_element.findall('mask'):
                label_name = mask_element.get('label')
                rle = mask_element.get('rle')
                left = int(mask_element.get('left'))
                top = int(mask_element.get('top'))
                width_mask = int(mask_element.get('width'))
                height_mask = int(mask_element.get('height'))
                annotation['labels'][label_name] = {
                    'rle': rle,
                    'left': left,
                    'top': top,
                    'width': width_mask,
                    'height': height_mask
                }
            img_files.append({'image_path': image_path, 'annotation': annotation})
        return img_files


class CervicalDataset(Dataset):
    def __init__(self, xml_path, class_names=('C2', 'C3', 'C4', 'C5', 'C6', 'C7'), mode='train',
                 transform=None):
        self.dataset = BaseDataset(xml_path)
        self.mode = mode
        self.transform = transform
        self.class_names = class_names
        self.class_map = {name: i for i, name in enumerate(self.class_names)}

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img_file, annotation_data = self.dataset.__getitem__(index)
        img = cv2.imread(img_file)
        if img is None:
            logger.warning(f"Could not read image {img_file}, returning random sample")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        height, width = img.shape[:2]
        masks = []
        labels = []
        boxes = []

        for label_name, mask_data in annotation_data['labels'].items():
            if label_name not in self.class_names:
                logger.debug(f"Skipping label {label_name} not in class_names")
                continue

            rle = mask_data['rle']
            logger.debug(f"Processing RLE for {label_name}: {rle}")
            mask = rle_decode(rle, (mask_data['height'], mask_data['width']))
            left = mask_data['left']
            top = mask_data['top']

            # Kiểm tra mask hợp lệ
            if mask.shape != (mask_data['height'], mask_data['width']):
                logger.warning(
                    f"Invalid mask shape {mask.shape} for {label_name}, expected {mask_data['height']}x{mask_data['width']}")
                continue
            if mask.sum() == 0:
                logger.warning(f"Empty mask for {label_name}")
                continue

            full_mask = np.zeros((height, width), dtype=np.uint8)
            try:
                full_mask[top:top + mask_data['height'], left:left + mask_data['width']] = mask
            except ValueError as e:
                logger.error(
                    f"Error placing mask for {label_name}: {e}, top={top}, left={left}, mask_shape={mask.shape}")
                continue

            y_indices, x_indices = np.where(full_mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                ymin, ymax = np.min(y_indices), np.max(y_indices)
                xmin, xmax = np.min(x_indices), np.max(x_indices)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    masks.append(full_mask)
                    labels.append(self.class_map[label_name])
                else:
                    logger.warning(f"Invalid bounding box for {label_name}: [{xmin}, {ymin}, {xmax}, {ymax}]")
            else:
                logger.warning(f"No valid pixels in mask for {label_name}")

        # Chuyển đổi sang tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = np.array(masks, dtype=np.uint8)  # Chuyển thành mảng NumPy 3D: (num_masks, H, W)

        # Kiểm tra hình dạng của masks
        if masks.size == 0:
            logger.warning(f"No valid masks for image {img_file}, creating empty masks")
            masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            logger.info(f"Masks shape before transform: {masks.shape}")

        if self.transform:
            try:
                transformed = self.transform(
                    image=img,
                    masks=masks,  # Truyền NumPy array 3D
                    bboxes=boxes.numpy(),
                    labels=labels.numpy()
                )
                img = transformed['image']
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                masks = np.array(transformed['masks'], dtype=np.uint8)  # Đảm bảo masks là NumPy 3D
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)

                # Kiểm tra và lọc các mask/bbox hợp lệ sau transform
                valid_indices = []
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    if box[2] > box[0] and box[3] > box[1] and mask.sum() > 0:
                        valid_indices.append(i)
                    else:
                        logger.warning(f"Invalid transformed mask/bbox at index {i}")

                if len(valid_indices) > 0:
                    boxes = boxes[valid_indices]
                    masks = masks[valid_indices]
                    labels = labels[valid_indices]
                else:
                    logger.warning(f"No valid masks/bboxes after transform for image {img_file}")
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    masks = np.zeros((0, height, width), dtype=np.uint8)
                    labels = torch.zeros((0,), dtype=torch.int64)
            except Exception as e:
                logger.error(f"Error applying transforms: {e}, returning empty sample")
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = np.zeros((0, height, width), dtype=np.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)

        # Chuyển masks thành tensor sau transform
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,),
                                                                                                                 dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return img, target


if __name__ == "__main__":
    pass