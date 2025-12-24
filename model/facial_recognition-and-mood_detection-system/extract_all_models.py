"""
Complete ONNX Model Extraction Script
Extracts ALL models (even pre-existing ONNX ones) to onnx_models directory
"""

import os
import sys
import shutil
import torch
import numpy as np
from pathlib import Path


def install_onnxscript():
    """Install onnxscript in the current environment"""
    print("📦 Installing onnxscript in current environment...")
    import subprocess

    # Use the current Python interpreter
    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, "-m", "pip", "install", "onnxscript"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ onnxscript installed successfully")
        return True
    else:
        print(f"❌ Failed to install onnxscript: {result.stderr}")
        return False


def find_uniface_model_path(model_name):
    """Find where UniFace stores its ONNX models"""
    try:
        import uniface
        uniface_path = Path(uniface.__file__).parent

        # Common locations
        possible_paths = [
            uniface_path / "weights",
            uniface_path / "models",
            uniface_path / "pretrained",
            uniface_path / ".." / "weights",
            Path.home() / ".uniface" / "weights",
            Path.home() / ".cache" / "uniface",
        ]

        print(f"\n🔍 Searching for UniFace models...")
        print(f"   UniFace package location: {uniface_path}")

        for path in possible_paths:
            if path.exists():
                print(f"   ✓ Found directory: {path}")
                # List .onnx files
                onnx_files = list(path.glob("**/*.onnx"))
                if onnx_files:
                    print(f"     Found {len(onnx_files)} ONNX files:")
                    for f in onnx_files[:5]:  # Show first 5
                        print(f"       - {f.name}")
                    return path

        print("   ⚠️  No pre-existing ONNX files found")
        return None

    except Exception as e:
        print(f"⚠️  Error searching for models: {e}")
        return None


def copy_arcface_model():
    """Copy or extract ArcFace ONNX model"""
    print("\n" + "=" * 60)
    print("1. Extracting ArcFace Model")
    print("=" * 60)

    try:
        from uniface import ArcFace

        arcface = ArcFace()

        # Check if model_path attribute exists
        if hasattr(arcface, 'model_path'):
            source_path = Path(arcface.model_path)
            print(f"✓ Found ArcFace model at: {source_path}")

            if source_path.exists():
                output_dir = Path("onnx_models")
                output_dir.mkdir(parents=True, exist_ok=True)
                dest_path = output_dir / "arcface_model.onnx"

                shutil.copy2(source_path, dest_path)
                print(f"✅ ArcFace model copied to: {dest_path}")
                print(f"   Size: {dest_path.stat().st_size / (1024*1024):.1f} MB")
                return True
            else:
                print(f"⚠️  Model file doesn't exist: {source_path}")

        # Fallback: search for it
        print("🔍 Searching for ArcFace model in UniFace cache...")
        uniface_dir = find_uniface_model_path("arcface")
        if uniface_dir:
            arcface_files = list(uniface_dir.glob("**/arcface*.onnx")) + \
                           list(uniface_dir.glob("**/w600k*.onnx"))

            if arcface_files:
                source_path = arcface_files[0]
                output_dir = Path("onnx_models")
                output_dir.mkdir(parents=True, exist_ok=True)
                dest_path = output_dir / "arcface_model.onnx"

                shutil.copy2(source_path, dest_path)
                print(f"✅ ArcFace model copied to: {dest_path}")
                return True

        print("⚠️  Could not locate ArcFace ONNX model")
        print("   (It's still being used internally by UniFace)")
        return False

    except Exception as e:
        print(f"❌ Failed to extract ArcFace: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_retinaface_model():
    """Copy or extract RetinaFace ONNX model"""
    print("\n" + "=" * 60)
    print("2. Extracting RetinaFace Model")
    print("=" * 60)

    try:
        from uniface import RetinaFace
        from uniface.constants import RetinaFaceWeights

        detector = RetinaFace(model_name=RetinaFaceWeights.MNET_V2)

        # Check common attributes where the model path might be stored
        possible_attrs = ['model_path', 'weights_path', 'onnx_path', 'weight_file', 'model_file']

        for attr in possible_attrs:
            if hasattr(detector, attr):
                attr_value = getattr(detector, attr)
                if attr_value and isinstance(attr_value, (str, Path)):
                    source_path = Path(attr_value)
                    print(f"✓ Found RetinaFace model path in '{attr}': {source_path}")

                    if source_path.exists():
                        output_dir = Path("onnx_models")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = output_dir / "retinaface_model.onnx"

                        shutil.copy2(source_path, dest_path)
                        print(f"✅ RetinaFace model copied to: {dest_path}")
                        print(f"   Size: {dest_path.stat().st_size / (1024*1024):.1f} MB")
                        return True

        # Advanced search: check .uniface directory
        print("🔍 Searching ~/.uniface/models/ directory...")
        uniface_models = Path.home() / ".uniface" / "models"
        if uniface_models.exists():
            print(f"   ✓ Found .uniface directory: {uniface_models}")
            retina_files = list(uniface_models.glob("**/retina*.onnx")) + \
                          list(uniface_models.glob("**/mnet*.onnx")) + \
                          list(uniface_models.glob("**/*det*.onnx"))

            if retina_files:
                print(f"   ✓ Found {len(retina_files)} potential RetinaFace models")
                source_path = retina_files[0]
                output_dir = Path("onnx_models")
                output_dir.mkdir(parents=True, exist_ok=True)
                dest_path = output_dir / "retinaface_model.onnx"

                shutil.copy2(source_path, dest_path)
                print(f"✅ RetinaFace model copied to: {dest_path}")
                print(f"   Size: {dest_path.stat().st_size / (1024*1024):.1f} MB")
                return True

        # Fallback: general search
        print("🔍 Searching for RetinaFace model in UniFace cache...")
        uniface_dir = find_uniface_model_path("retinaface")
        if uniface_dir:
            retina_files = list(uniface_dir.glob("**/retina*.onnx")) + \
                          list(uniface_dir.glob("**/mnet*.onnx"))

            if retina_files:
                source_path = retina_files[0]
                output_dir = Path("onnx_models")
                output_dir.mkdir(parents=True, exist_ok=True)
                dest_path = output_dir / "retinaface_model.onnx"

                shutil.copy2(source_path, dest_path)
                print(f"✅ RetinaFace model copied to: {dest_path}")
                return True

        print("⚠️  Could not locate RetinaFace ONNX model")
        print("   (It's still being used internally by UniFace)")
        print("   This is OK - your system will work fine with PyTorch mode!")
        return False

    except Exception as e:
        print(f"❌ Failed to extract RetinaFace: {e}")
        import traceback
        traceback.print_exc()
        return False

"""

def convert_emotion_model():
   ' Convert Emotion model to ONNX'
    print("\n" + "=" * 60)
    print("3. Converting Emotion Model")
    print("=" * 60)

    # First, ensure onnxscript is installed
    try:
        import onnxscript
        print("✓ onnxscript is available")
    except ImportError:
        print("⚠️  onnxscript not found in current environment")
        if not install_onnxscript():
            print("❌ Failed to install onnxscript")
            print("\n🔧 Manual fix:")
            print(f"   Run: {sys.executable} -m pip install onnxscript")
            return False

        # Try importing again
        try:
            import onnxscript
            print("✓ onnxscript now available")
        except ImportError:
            print("❌ Still can't import onnxscript")
            print("   You may need to restart Python after installation")
            return False

    try:
        from uniface import Emotion
        import torch.onnx.utils
        from torch.onnx import OperatorExportTypes

        emotion = Emotion()
        model = emotion.model
        model.eval()

        print(f"✓ Emotion model loaded: {type(model)}")

        output_dir = Path("onnx_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "emotion_model.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, 3, 112, 112)

        print("✓ Exporting to ONNX...")
        print("  Using TorchScript-compatible ONNX export...")

        # For TorchScript models, export directly without dynamic_axes
        # which causes issues with the new PyTorch ONNX exporter
        with torch.no_grad():
            # Export without dynamic_axes to avoid TorchScript issues
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )

        print(f"✅ Emotion model saved to: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Input shape: (batch, 3, 112, 112)")
        print(f"   Output shape: (batch, {len(emotion.emotion_labels)})")

        # Verify
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model is valid")
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")

        # Test inference
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(output_path))
            test_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})
            print(f"✅ Inference test passed")
            print(f"   Output shape: {output[0].shape}")
        except Exception as e:
            print(f"⚠️  Inference test warning: {e}")

        return True

    except Exception as e:
        print(f"❌ Failed to convert Emotion: {e}")
        import traceback
        traceback.print_exc()
        return False
"""

def main():
    """Main extraction script"""
    print("\n" + "=" * 60)
    print("🚀 COMPLETE ONNX MODEL EXTRACTION")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Copy ArcFace ONNX model (if found)")
    print("  2. Copy RetinaFace ONNX model (if found)")
    print("  3. Convert Emotion model to ONNX")
    print("\nAll models will be saved to: onnx_models/")
    print("=" * 60)

    input("\nPress Enter to continue...")

    # Create output directory
    output_dir = Path("onnx_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
        #"Emotion": convert_emotion_model(),
    # Extract all models
    results = {
        "ArcFace": copy_arcface_model(),
        "RetinaFace": copy_retinaface_model(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)

    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "⚠️  PARTIAL/FAILED"
        print(f"{model_name}: {status}")

    print("\n" + "=" * 60)
    print("📁 Output Directory:")
    print("=" * 60)

    if output_dir.exists():
        onnx_files = list(output_dir.glob("*.onnx"))
        if onnx_files:
            print(f"\n✅ Found {len(onnx_files)} ONNX models in {output_dir}:")
            for f in onnx_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   • {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"\n⚠️  No ONNX files found in {output_dir}")

    print("\n" + "=" * 60)
    print("💡 IMPORTANT NOTES:")
    print("=" * 60)
    print("\n1. If ArcFace/RetinaFace extraction failed:")
    print("   → UniFace is still using them internally (optimized)")
    print("   → Your system will work fine with PyTorch mode (option 1)")
    print("\n2. If Emotion conversion failed:")
    print("   → Try manually: pip install onnxscript")
    print("   → Or use PyTorch mode (Emotion is already TorchScript)")
    print("\n3. To use extracted models:")
    print("   → Run: python face_recognition_emotion_system.py")
    print("   → Select option '2' for ONNX mode")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
