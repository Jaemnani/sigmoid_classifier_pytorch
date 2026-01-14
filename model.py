
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Build layers matches the original build() logic
        # Input shape in PyTorch is (C, H, W)
        
        self.layers = nn.ModuleList()
        
        # Conv 16
        self.conv1 = self._make_conv(input_shape[0], 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2) if self._is_stride_over(2) else nn.Identity()
        self.drop1 = nn.Dropout(0.1)
        
        # Conv 32
        self.conv2 = self._make_conv(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2) if self._is_stride_over(4) else nn.Identity()
        self.drop2 = nn.Dropout(0.15)
        
        # Conv 64
        self.conv3 = self._make_conv(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2) if self._is_stride_over(8) else nn.Identity()
        self.drop3 = nn.Dropout(0.2)
        
        # Conv 128
        self.conv4 = self._make_conv(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2, 2) if self._is_stride_over(16) else nn.Identity()
        self.drop4 = nn.Dropout(0.25)
        
        # Conv 256 (CAM activation)
        self.conv5 = self._make_conv(128, 256, 3)
        self.pool5 = nn.MaxPool2d(2, 2) if self._is_stride_over(32) else nn.Identity()
        self.drop5 = nn.Dropout(0.3)
        
        # Conv 256
        self.conv6 = self._make_conv(256, 256, 3)
        
        # Classification Layer
        # Original was Conv2D(kernel=1) -> GAP.
        # Equivalent to Conv2d 1x1
        self.classifier_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        nn.init.xavier_normal_(self.classifier_conv.weight) # glorot_normal
        if self.classifier_conv.bias is not None:
             nn.init.zeros_(self.classifier_conv.bias)

    def _is_stride_over(self, stride):
        # input_shape is (C, H, W)
        return self.input_shape[1] >= stride and self.input_shape[2] >= stride

    def _make_conv(self, in_channels, out_channels, kernel_size, bn=False):
        # Original: Conv2D(args), if bn: BatchNormalization, Activation('relu')
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=not bn)
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu') # he_normal
        if not bn:
            # Manually init bias if used
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        
        layers.append(conv)
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        x = self.conv5(x) # This is the CAM activation layer in original
        # We can hook this output if needed for CAM
        
        x = self.pool5(x)
        x = self.drop5(x)
        
        x = self.conv6(x)
        
        x = self.classifier_conv(x)
        x = torch.sigmoid(x) # Activation 'sigmoid' in original classification layer
        
        # Global Average Pooling
        # x is (B, NumClasses, H, W) -> (B, NumClasses)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        return x
