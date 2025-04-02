import torch
import hydra
import numpy as np
import cv2
import os
from omegaconf import DictConfig
from typing import Optional, List, Dict
from lightning import LightningDataModule
from src.model.model_module import model_module 
from src.model.mask_rcnn import MaskRCNN  

def visualize_predictions(image: np.ndarray, predictions: List[Dict], class_names: List[str]):
    """Visualizes the predicted bounding boxes, masks, and labels on the image."""
    image = image.copy()
    H, W, _ = image.shape
    for prediction in predictions:
        boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy()

        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if score > 0.5: # Adjustible
                x1, y1, x2, y2 = box
                color = (0, 255, 0)  
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                mask = (mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                mask_colored = np.uint8(mask_resized[:, :, None] * np.array([0, 0, 255]))
                roi = image[y1:y2, x1:x2]
                masked_roi = cv2.addWeighted(roi, 0.7, mask_colored, 0.3, 0)
                image[y1:y2, x1:x2] = masked_roi

                class_name = class_names[label]
                text = f"{class_name}: {score:.2f}"
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="data")
def main(cfg: DictConfig) -> Optional[float]:
    ckpt_path = 'path/to/your/mask_rcnn_checkpoint.ckpt' # Change this to where the checkpoint is pointed to after training
    num_classes = cfg.get("num_classes", 7) 
    class_names = cfg.get("class_names", ["background", "C2", "C2_lower", "C3", "C4", "C5", "C6", "C7"])


    model = model_module.load_from_checkpoint(
        net=MaskRCNN(num_classes=num_classes, backbone_pretrained=False), 
        checkpoint_path=ckpt_path,
        num_classes=num_classes 
    )
    model.eval()
    model.cuda()

    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    os.makedirs("inference", exist_ok=True)

    for idx, batch in enumerate(test_loader):
        images, _ = batch 
        original_images = [img.permute(1, 2, 0).cpu().numpy() * 255 for img in images]

        with torch.no_grad():
            predictions = model(torch.stack(images).cuda())

        for i, preds in enumerate(predictions):
            img_np = original_images[i].astype(np.uint8).copy()
            visualized_img = visualize_predictions(img_np, [preds], class_names)
            cv2.imwrite(f"inference/test_{idx}_{i}.png", visualized_img)

if __name__ == "__main__":
    main()
 