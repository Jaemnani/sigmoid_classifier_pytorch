
import os
import cv2
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

from model import Model
from eta import ETACalculator
from live_plot import LivePlot
from dataset import CustomDataset
from lr_scheduler import LRScheduler
from ace import AdaptiveCrossentropy
from ckpt_manager import CheckpointManager

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# print(f'Device : {device}')

class SigmoidClassifier(CheckpointManager):
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 model_name,
                 input_shape,
                 lr,
                 lrf,
                 alpha,
                 gamma,
                 warm_up,
                 momentum,
                 batch_size,
                 iterations,
                 label_smoothing,
                 aug_brightness,
                 aug_contrast,
                 aug_rotate,
                 aug_h_flip,
                 lr_policy='step',
                 checkpoint_interval=0,
                 show_class_activation_map=False,
                 show_live_plot=False,
                 cam_activation_layer_name='cam_activation',
                 last_conv_layer_name='squeeze_conv',
                 early_stopping_patience=10,
                 architecture='original'):
        super().__init__()
        self.input_shape = input_shape # (H, W, C)
        self.lr = lr
        self.lrf = lrf
        self.warm_up = warm_up
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr_policy = lr_policy 
        self.show_class_activation_map = show_class_activation_map
        self.show_live_plot = show_live_plot
        self.cam_activation_layer_name = cam_activation_layer_name
        self.last_conv_layer_name = last_conv_layer_name
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_patience = early_stopping_patience
        self.pretrained_iteration_count = 0
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_rotate = aug_rotate
        self.aug_h_flip = aug_h_flip
        self.architecture = architecture
        self.log_file = None
        warnings.filterwarnings(action='ignore')
        self.set_model_name(model_name)
        if self.checkpoint_interval == 0:
            self.checkpoint_interval = self.iterations

        train_image_path = self.unify_path(train_image_path)
        validation_image_path = self.unify_path(validation_image_path)

        self.train_image_paths, train_class_names, _ = self.init_image_paths(train_image_path)
        self.validation_image_paths, validation_class_names, self.include_unknown = self.init_image_paths(validation_image_path)
        

        if len(self.train_image_paths) == 0:
            print(f'no images in train_image_path : {train_image_path}')
            exit(0)
        if len(self.validation_image_paths) == 0:
            print(f'no images in validation_image_path : {validation_image_path}')
            exit(0)

        self.class_names = train_class_names
        
        # Datasets
        self.train_dataset = CustomDataset(
            root_path=train_image_path,
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            class_names=train_class_names,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast,
            aug_rotate=aug_rotate,
            aug_h_flip=aug_h_flip,
            is_training=True
        )
        
        self.validation_dataset = CustomDataset(
            root_path=validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            class_names=self.class_names,
            is_training=False
        )

        # PyTorch DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        self.iterations_per_epoch = len(self.train_loader)
        self.checkpoint_interval = self.iterations_per_epoch
        print(f'Set checkpoint interval to 1 epoch : {self.checkpoint_interval} iterations')
        
        # Model
        # Input shape in model is expected as (C, H, W) mostly for internals, but constructor takes logic from original which used (H,W,C).
        # We'll pass (C, H, W) to model constructor just to be safe if it needs channel info 
        model_input_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])
        self.model = Model(input_shape=model_input_shape, num_classes=len(self.class_names), architecture=self.architecture).to(device)
        self.feature_map = None
        self.model.classifier_conv.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.feature_map = output

    def init_logger(self, path):
        self.log_file = open(path, 'w')

    def log(self, msg, end='\n'):
        print(msg, end=end)
        if self.log_file:
            self.log_file.write(msg + end)
            self.log_file.flush()

    def load_model(self, model_path):
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(model_path)
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f'Loaded model from {model_path}')
        else:
            print(f'pretrained model not found : {model_path}')
            exit(0)

    def unify_path(self, path):
        if path == '':
            return path
        path = path.replace('\\', '/')
        if path.endswith('/'):
            path = path[:-1]
        return path

    def init_image_paths(self, image_path):
        include_unknown = False
        dir_paths = sorted(glob(f'{image_path}/*'))
        for i in range(len(dir_paths)):
            dir_paths[i] = dir_paths[i].replace('\\', '/')
        image_paths = []
        class_counts = []
        class_name_set = set()
        unknown_class_count = 0
        print('class image count')
        for dir_path in dir_paths:
            if not os.path.isdir(dir_path):
                continue
            dir_name = dir_path.split('/')[-1]
            if dir_name[0] == '_':
                print(f'class dir {dir_name} is ignored. dir_name[0] == "_"')
                continue
            if dir_name == 'unknown':
                include_unknown = True
            else:
                class_name_set.add(dir_name)
            cur_class_image_paths = glob(f'{dir_path}/**/*.jpg', recursive=True)
            for i in range(len(cur_class_image_paths)):
                cur_class_image_paths[i] = cur_class_image_paths[i].replace('\\', '/')
            image_paths += cur_class_image_paths
            cur_class_image_count = len(cur_class_image_paths)
            if dir_name == 'unknown':
                unknown_class_count = cur_class_image_count
            else:
                class_counts.append(cur_class_image_count)
            print(f'class {dir_name} : {cur_class_image_count}')
        print()
        class_names = sorted(list(class_name_set))
        return image_paths, class_names, include_unknown

    def draw_cam(self, x, label):
        self.model.eval()
        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(device)
            _ = self.model(x_tensor)

        # feature_map shape: (1, C, H, W)
        # Using the direct output of classifier_conv as CAM
        # This matches the "weights * features" logic since classifier_conv does exactly that.
        cam_map = self.feature_map.cpu().numpy()[0] # (C, H, W)
        
        img_h, img_w, img_c = self.input_shape
        x_np = x.permute(1, 2, 0).cpu().numpy() # (C, H, W) -> (H, W, C)
        
        if img_c == 1:
            x_np = np.concatenate([x_np, x_np, x_np], axis=-1)
        
        # Denormalize roughly for visualization if needed. 
        # Assuming input was 0-1.
        
        image_grid = None
        for idx, cls in enumerate(self.class_names):
            # cam_map[idx] is the heatmap for class idx
            cam = cam_map[idx] # (H, W) (small size)
            
            # Normalize CAM
            cam -= np.min(cam)
            if np.max(cam) > 0:
                cam /= np.max(cam)
            cam *= 255.0
            cam = cam.astype(np.uint8)
            
            # Resize to image size
            cam = cv2.resize(cam, (img_w, img_h))
            
            # Heatmap color
            cam_jet = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            
            # Original image (0-1) -> 0-255
            org_img_uint8 = (x_np * 255.0).astype(np.uint8)
            if img_c == 1 and org_img_uint8.shape[-1] == 1:
                 org_img_uint8 = cv2.cvtColor(org_img_uint8, cv2.COLOR_GRAY2BGR)
            elif img_c == 3:
                 # PyTorch is RGB, OpenCV is BGR
                 org_img_uint8 = cv2.cvtColor(org_img_uint8, cv2.COLOR_RGB2BGR)

            # Blend
            cam_blended = cv2.addWeighted(org_img_uint8, 0.6, cam_jet, 0.4, 0)
            
            # Label box logic (simplified)
            # Create a small bar indicating if this is the label
            label_indicator = np.zeros((img_h, 10, 3), dtype=np.uint8)
            if idx == np.argmax(label.cpu().numpy()):
                label_indicator[:] = (0, 255, 0) # Green for true label
            
            # Concatenate: Indicator | Original | Heatmap | Blended
            row = np.hstack([label_indicator, org_img_uint8, cam_jet, cam_blended])
            
            if image_grid is None:
                image_grid = row
            else:
                image_grid = np.vstack([image_grid, row])
                
        # Resize for display if too tall
        max_h = 800
        if image_grid.shape[0] > max_h:
            ratio = max_h / image_grid.shape[0]
            new_w = int(image_grid.shape[1] * ratio)
            image_grid = cv2.resize(image_grid, (new_w, max_h))
            
        cv2.imshow('CAM', image_grid)
        cv2.waitKey(1)
        
        self.model.train()

    def print_loss(self, progress_str, loss, val_loss=None):
        if val_loss is not None and not np.isnan(val_loss):
            self.log(f'\r{progress_str} loss => {loss:.4f}, val_loss => {val_loss:.4f}', end='')
        else:
            self.log(f'\r{progress_str} loss => {loss:.4f}', end='')

    def compute_validation_loss(self, loss_function):
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in self.validation_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_pred = self.model(batch_x)
                loss = loss_function(batch_y, y_pred)
                total_loss += loss.item()
                count += 1
        self.model.train()
        return total_loss / count if count > 0 else 0.0

    def train(self):
        if self.pretrained_iteration_count >= self.iterations:
            self.log(f'pretrained iteration count {self.pretrained_iteration_count} is greater or equal than target iterations {self.iterations}')
            exit(0)

        self.log(f'\ntrain on {len(self.train_image_paths)} samples')
        self.log(f'validate on {len(self.validation_image_paths)} samples\n')
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.momentum, 0.999))
        
        # Loss
        loss_function = AdaptiveCrossentropy(alpha=self.alpha, gamma=self.gamma, label_smoothing=self.label_smoothing, reduce='sum').to(device)
        
        # Scheduler
        lr_scheduler = LRScheduler(lr=self.lr, lrf=self.lrf, iterations=self.iterations, warm_up=self.warm_up, policy=self.lr_policy)
        
        self.checkpoint_dir = self.init_checkpoint_dir()
        self.init_logger(f'{self.checkpoint_dir}/train_log.txt')
        
        # Save class names
        with open(f'{self.checkpoint_dir}/class_names.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f'{class_name}\n')
        
        self.log(f'\n{self.model_name}_config')
        self.log(f'input_shape : {self.input_shape}')
        self.log(f'lr : {self.lr}')
        self.log(f'lrf : {self.lrf}')
        self.log(f'alpha : {self.alpha}')
        self.log(f'gamma : {self.gamma}')
        self.log(f'warm_up : {self.warm_up}')
        self.log(f'momentum : {self.momentum}')
        self.log(f'batch_size : {self.batch_size}')
        self.log(f'iterations : {self.iterations}')
        self.log(f'label_smoothing : {self.label_smoothing}')
        self.log(f'aug_brightness : {self.aug_brightness}')
        self.log(f'aug_contrast : {self.aug_contrast}')
        self.log(f'aug_rotate : {self.aug_rotate}')
        self.log(f'aug_h_flip : {self.aug_h_flip}')
        self.log(f'checkpoint_interval : {self.checkpoint_interval}')
        self.log(f'early_stopping_patience : {self.early_stopping_patience}')
        self.log(f'show_class_activation_map : {self.show_class_activation_map}')
        self.log(f'show_live_plot : {self.show_live_plot}')
        
        iteration_count = self.pretrained_iteration_count
        eta_calculator = ETACalculator(iterations=self.iterations, start_iteration=iteration_count)
        eta_calculator.start()

        if self.show_live_plot:
            live_plot = LivePlot(iterations=self.iterations, legends=['train_loss', 'val_loss'])
        else:
            live_plot = LivePlot(iterations=self.iterations, legends=['train_loss', 'val_loss'], output_file=f'{self.checkpoint_dir}/loss_graph.png')
        
        
        current_val_loss = np.nan
        best_acc = 0.0
        patience_count = 0
        
        self.model.train()
        
        # Pre-fetch fixed validation batch for live plotting (use shuffle=True to get representative batch)
        # We create a temporary loader just for this initialization
        temp_val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_iter = iter(temp_val_loader)
        try:
            fixed_val_x, fixed_val_y = next(val_iter)
        except StopIteration:
            fixed_val_x, fixed_val_y = None, None
            print("Warning: Validation dataset is empty.")
        del temp_val_loader

        if fixed_val_x is not None:
            fixed_val_x = fixed_val_x.to(device)
            fixed_val_y = fixed_val_y.to(device)

        early_stopping_triggered = False
        while True:
            for batch_x, batch_y in self.train_loader:
                if iteration_count >= self.iterations:
                    break

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                lr_scheduler.update(optimizer, iteration_count)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = loss_function(batch_y, y_pred)
                loss.backward()
                optimizer.step()
                
                if self.show_class_activation_map and iteration_count % 100 == 0:
                    self.draw_cam(batch_x[0], batch_y[0])
                
                # Calculate validation loss for current batch (using fixed batch for speed)
                if fixed_val_x is not None:
                    with torch.no_grad():
                        self.model.eval()
                        val_pred = self.model(fixed_val_x)
                        current_val_loss = loss_function(fixed_val_y, val_pred).item()
                        self.model.train()

                iteration_count += 1
                progress_str = eta_calculator.update(iteration_count)
                self.print_loss(progress_str, loss.item(), val_loss=current_val_loss)
                
                live_plot.update(train_loss=loss.item(), val_loss=current_val_loss)

                if iteration_count % 2000 == 0:
                    self.save_last_model(self.model.state_dict(), iteration_count)
                    
                if iteration_count >= int(self.iterations * self.warm_up) and iteration_count % self.checkpoint_interval == 0:
                    acc, class_score, unknown_score = self.evaluate()
                    self.model.train() # Set back to train mode
                    content = f'_acc_{acc:.4f}_class_score_{class_score:.4f}'
                    if self.include_unknown:
                        content += f'_unknown_score_{unknown_score:.4f}'
                    
                    if acc > best_acc:
                        best_acc = acc
                        patience_count = 0
                        self.save_best_model(self.model.state_dict(), iteration_count, metric=acc, content=content)
                    else:
                        patience_count += 1
                        self.log(f'early stopping patience count : {patience_count}/{self.early_stopping_patience}')
                        if self.early_stopping_patience > 0 and patience_count >= self.early_stopping_patience:
                            self.log(f'\nearly stopping triggered at iteration {iteration_count}')
                            early_stopping_triggered = True
                            break
            
            if iteration_count >= self.iterations or early_stopping_triggered:
                self.log('\ntrain end successfully')
                break

    def evaluate(self, dataset='validation', unknown_threshold=0.5):
        self.model.eval()
        loader = self.validation_loader if dataset == 'validation' else self.train_loader
        # Create a separate loader for evaluation if needed to avoid messing up training state logic, 
        # but standard loaders are fine. For exact parity with original 'one_batch' logic we might want batch_size=1,
        # but regular batch size is faster and equivalent for metrics.
        
        # However, to match the original line-by-line printing style and logic, let's iterate carefully.
        # Original: batch_size=1
        
        eval_loader = DataLoader(
            self.validation_dataset if dataset == 'validation' else self.train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        self.log(f'')
        num_classes = self.model.num_classes
        hit_class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        total_class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        hit_class_score_sums = np.zeros(shape=(num_classes,), dtype=np.float32)
        hit_unknown_count = 0
        total_unknown_count = 0
        hit_unknown_score_sum = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(eval_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                y_pred = self.model(batch_x)[0] # (1, C) -> (C)
                y_pred = y_pred.cpu().numpy()
                batch_y = batch_y.cpu().numpy()[0]
                
                max_score_index = np.argmax(y_pred)
                max_score = y_pred[max_score_index]
                
                if np.sum(batch_y) == 0.0:  # case unknown
                    total_unknown_count += 1
                    if max_score < unknown_threshold:
                        hit_unknown_count += 1
                        hit_unknown_score_sum += max_score
                else:  # case classification
                    true_class_index = np.argmax(batch_y)
                    total_class_counts[true_class_index] += 1
                    if max_score_index == true_class_index:
                        if self.include_unknown:
                            if max_score >= unknown_threshold:
                                hit_class_counts[true_class_index] += 1
                                hit_class_score_sums[true_class_index] += max_score
                        else:
                            hit_class_counts[true_class_index] += 1
                            hit_class_score_sums[true_class_index] += max_score

        total_acc_sum = 0.0
        class_score_sum = 0.0
        
        self.log(f'')
        for i in range(len(total_class_counts)):
            cur_class_acc = hit_class_counts[i] / (float(total_class_counts[i]) + 1e-5)
            cur_class_score = hit_class_score_sums[i] / (float(hit_class_counts[i]) + 1e-5)
            total_acc_sum += cur_class_acc
            class_score_sum += cur_class_score
            self.log(f'[class {i:2d}] acc : {cur_class_acc:.4f}, score : {cur_class_score:.4f}')

        valid_class_count = num_classes
        unknown_score = 0.0
        if self.include_unknown and total_unknown_count > 0:
            unknown_acc = hit_unknown_count / float(total_unknown_count + 1e-5)
            unknown_score = hit_unknown_score_sum / float(hit_unknown_count + 1e-5)
            total_acc_sum += unknown_acc
            valid_class_count += 1
            self.log(f'[class unknown] acc : {unknown_acc:.4f}, score : {unknown_score:.4f}')

        class_acc = total_acc_sum / valid_class_count
        class_score = class_score_sum / num_classes
        if self.include_unknown:
            self.log(f'total accuracy with unknown threshold({unknown_threshold:.2f}) : {class_acc:.4f}, class_score : {class_score:.4f}, unknown_score : {unknown_score:.4f}\n')
        else:
            self.log(f'total accuracy : {class_acc:.4f}, class_score : {class_score:.4f}\n')
            
        return class_acc, class_score, unknown_score
