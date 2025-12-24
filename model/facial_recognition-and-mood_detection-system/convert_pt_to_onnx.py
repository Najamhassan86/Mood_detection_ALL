import torch
import torch.onnx
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the TorchScript model
model_path = os.path.join(script_dir, "emotion_model.pt")
model = torch.jit.load(model_path)

# Example dummy input, adjust according to your model's expected input size
dummy_input = torch.randn(1, 3, 112, 112)  # Input size: (batch_size, channels, height, width)

# Define the ONNX output file path
onnx_model_path = os.path.join(script_dir, "emotion_model.onnx")

# Export the TorchScript model to ONNX using legacy exporter
torch.onnx.export(
    model,                        # The model to be exported
    dummy_input,                  # Example input tensor
    onnx_model_path,              # Path to save the ONNX model
    export_params=True,           # Export trained parameters
    opset_version=17,             # ONNX version (17 is recommended for 2024+)
    do_constant_folding=True,     # Enable constant folding for optimization
    input_names=['input'],        # Input tensor names
    output_names=['output'],      # Output tensor names
    verbose=True,                 # Optionally display the conversion details
    dynamo=False                  # Use legacy TorchScript-based exporter
)

print(f"Model successfully converted to ONNX: {onnx_model_path}")
