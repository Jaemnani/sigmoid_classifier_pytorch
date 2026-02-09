import torch
import torch.onnx
from model import Model
import os
import onnx

import torch
import torch.onnx
from model import Model
import os
import onnx
import argparse

def convert_to_onnx(model_path, input_shape, num_classes, architecture):
    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    output_path = os.path.splitext(model_path)[0] + '.onnx'

    # Initialize model
    print(f"Initializing model with input_shape={input_shape}, num_classes={num_classes}, architecture={architecture}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model expects input_shape as tuple (C, H, W)
    # If user provides H W C, we might need to adjust, but let's assume valid tuple input or handle string parsing carefully.
    # The Model class takes input_shape as is. In the training script, it was passing (1, 96, 96).
    
    model = Model(input_shape=input_shape, num_classes=num_classes, architecture=architecture).to(device)
    
    # Load weights
    print(f"Loading weights...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Export
    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(model,               # model being run
                          dummy_input,         # model input
                          output_path,         # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,    # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the model's input names
                          output_names = ['output'], # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
        print("Export successful.")
    except Exception as e:
        print(f"Error during export: {e}")
        return

    # Verify ONNX model
    print("Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, 
                        required=True, help='Path to the .pt model file')
    parser.add_argument('--input_shape', type=int, nargs=3, 
                        default=[1, 96, 96], help='Input shape (C, H, W). Default: 1 96 96')
    parser.add_argument('--num_classes', type=int, 
                        default=3, help='Number of classes. Default: 3')
    parser.add_argument('--architecture', type=str, 
                        default='original', help='Model architecture. Default: original')

    args = parser.parse_args()
    
    # input_shape from nargs will be a list, convert to tuple
    input_shape = tuple(args.input_shape)
    
    convert_to_onnx(args.model_path, input_shape, args.num_classes, args.architecture)
