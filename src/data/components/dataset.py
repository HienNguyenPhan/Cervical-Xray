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

def rle_decode(rle, shape):
    """Decodes COCO RLE encoded mask."""
    try:
        if isinstance(rle, str):
            rle = mask_utils.frPyObjects({'counts': rle, 'size': shape}, shape[0], shape[1])
        return mask_utils.decode(rle).reshape(shape).astype(np.uint8)
    except Exception as e:
        print(f"Error decoding RLE: {e}")
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
    def __init__(self, xml_path, class_names=('C2', 'C2_lower', 'C3', 'C4', 'C5', 'C6', 'C7'), mode='train', transform=None):
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
        height, width = img.shape[:2]
        masks = []
        labels = []
        boxes = []

        for label_name, mask_data in annotation_data['labels'].items():
            if label_name in self.class_names:
                rle = mask_data['rle']
                mask = rle_decode({'size': [mask_data['height'], mask_data['width']], 'counts': rle},
                                  (mask_data['height'], mask_data['width']))
                left = mask_data['left']
                top = mask_data['top']

                # Pad the mask to the original image size
                full_mask = np.zeros((height, width), dtype=np.uint8)
                full_mask[top:top + mask_data['height'], left:left + mask_data['width']] = mask

                # Extract bounding box from the mask
                y_indices, x_indices = np.where(full_mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    ymin, ymax = np.min(y_indices), np.max(y_indices)
                    xmin, xmax = np.min(x_indices), np.max(x_indices)
                    # Đảm bảo box hợp lệ
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        masks.append(full_mask)
                        labels.append(self.class_map[label_name])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        if self.transform:
            transformed = self.transform(image=img, masks=masks.numpy(), bboxes=boxes.numpy(), labels=labels.numpy())
            img = transformed['image']
            masks = torch.as_tensor(transformed['masks'], dtype=torch.uint8)
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)

            valid_indices = []
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                if box[2] > box[0] and box[3] > box[1] and mask.sum() > 0:
                    valid_indices.append(i)

            if len(valid_indices) > 0:
                boxes = boxes[valid_indices]
                masks = masks[valid_indices]
                labels = labels[valid_indices]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((0, height, width), dtype=torch.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks

        return img, target

if __name__ == "__main__":
    pass