
import os
import cv2
import numpy as np
import shutil as sh
import torch
from concurrent.futures import ThreadPoolExecutor

from glob import glob
from tqdm import tqdm
from model import Model

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')




def letterbox(img, new_shape=(64, 64), color=(0, 0, 0)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Ensure dimensions are at least 1x1
    new_unpad = (max(1, new_unpad[0]), max(1, new_unpad[1]))
    
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Handle grayscale value
    if len(img.shape) == 2:
            value = color[0] if isinstance(color, tuple) else color
    else:
            value = color
            
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return img

def load_x_image_path(image_path, color_mode, input_size, input_shape):
    data = np.fromfile(image_path, dtype=np.uint8)
    x = cv2.imdecode(data, color_mode)
    x = letterbox(x, new_shape=(input_size[1], input_size[0]))
    if input_shape[0] == 3: # input_shape is (C, H, W)
         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # swap rb
    
    # Preprocess for PyTorch: (H, W, C) -> (C, H, W) and normalize
    # If grayscale, need to ensure channel dim exists if it was dropped
    if input_shape[0] == 1 and len(x.shape) == 2:
        x = x[..., np.newaxis]
        
    x = np.asarray(x).astype('float32') / 255.0
    x = np.transpose(x, (2, 0, 1)) # HWC -> CHW
    x = x[np.newaxis, ...] # Add batch dim -> (1, C, H, W)
    
    return torch.from_numpy(x), image_path


def auto_classification(model_path, image_path, save_path, num_classes=1000, input_shape=(3, 64, 64), architecture='original', save_with_score=False, unknown_threshold=0.5):
    # Note: Model architecture needs to be known to instantiate Model class.
    # In TensorFlow metadata was saved in the h5, in PyTorch usually we need to know the args.
    # We will assume standard args or add them as parameters.
    # For now defaulting to what seems to be used (imagenet 1000 classes maybe?) or user needs to supply it. 
    # The original script loaded model directly.
    
    model = Model(input_shape=input_shape, num_classes=num_classes, architecture=architecture).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # input_shape from model instance
    # model.input_shape is (C, H, W)
    c, h, w = model.input_shape
    input_size = (w, h)
    color_mode = cv2.IMREAD_GRAYSCALE if c == 1 else cv2.IMREAD_COLOR
    
    image_path = image_path.replace('\\', '/')
    image_paths = glob(f'{image_path}/*.jpg')
    pool = ThreadPoolExecutor(8)
    fs = []
    for img_path in image_paths:
        fs.append(pool.submit(load_x_image_path, img_path, color_mode, input_size, model.input_shape))

    with torch.no_grad():
        for f in tqdm(fs):
            x, img_path = f.result()
            x = x.to(device)
            # y = torch.softmax(y, dim=0).cpu().numpy() # Add softmax if model output is logits (original used sigmoid, our model uses sigmoid)
            # Wait, `model.py` uses sigmoid at the end. So outputs are already 0-1.
            # However `auto_classification` used `predict_on_batch` which returns model output.
            y = model(x)[0].cpu().numpy()
            
            class_index = np.argmax(y)
            score = y[class_index]
            
            score_dir = ''
            if save_with_score:
                score_dir = 'under_10'
                if score > 0.9:
                    score_dir = 'over_90'
                elif score > 0.8:
                    score_dir = 'over_80'
                elif score > 0.7:
                    score_dir = 'over_70'
                elif score > 0.6:
                    score_dir = 'over_60'
                elif score > 0.5:
                    score_dir = 'over_50'
                elif score > 0.4:
                    score_dir = 'over_40'
                elif score > 0.3:
                    score_dir = 'over_30'
                elif score > 0.2:
                    score_dir = 'over_20'
                elif score > 0.1:
                    score_dir = 'over_10'

            if score < unknown_threshold:
                save_dir = f'{save_path}/unknown'
                if save_with_score:
                    save_dir += f'/{score_dir}'
                os.makedirs(save_dir, exist_ok=True)
                sh.copy2(img_path, save_dir)
            else:
                save_dir = f'{save_path}/{class_index}'
                if save_with_score:
                    save_dir += f'/{score_dir}'
                os.makedirs(save_dir, exist_ok=True)
                sh.copy2(img_path, save_dir)


def main():
    # model_path = r'results/train/hd_ir_6/best_17548_iter_acc_0.6519_class_score_0.7163_unknown_score_0.1461.pt'
    model_path = r'results/train/hd_ir_7/best_23534_iter_acc_0.8594_class_score_0.8386_unknown_score_0.1309.pt'
    img_path = r'./test_data/test_fp_05_drop'
    

    save_with_score = False

    unknown_threshold = 0.45
    # unknown_threshold = 0.43

    # Create distinct results directory
    input_basename = os.path.basename(img_path.rstrip('/\\'))
    results_base = 'results/auto_classification'
    os.makedirs(results_base, exist_ok=True)
    
    n = 0
    while True:
        save_path = os.path.join(results_base, f'{input_basename}_{n}')
        if not os.path.exists(save_path):
            break
        n += 1
    
    # These parameters need to match training
    # hd_ir_6: efficient, 64x64
    # hd_ir_7: original, 96x96
    
    # auto_classification(model_path, img_path, save_path, num_classes=3, input_shape=(1, 64, 64), architecture='efficient', save_with_score=save_with_score, unknown_threshold=unknown_threshold)
    auto_classification(model_path, img_path, save_path, num_classes=3, input_shape=(1, 96, 96), architecture='original', save_with_score=save_with_score, unknown_threshold=unknown_threshold)
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    main()
