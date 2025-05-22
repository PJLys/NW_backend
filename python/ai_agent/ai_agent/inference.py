import numpy as np
import tflite_runtime.interpreter as tflite
import os


class TFLiteModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded. Input: {self.input_details}, Output: {self.output_details}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        expected_shape = self.input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            raise ValueError(f"Expected input shape {expected_shape}, got {input_data.shape}")
        
        input_data = input_data.astype(self.input_details[0]['dtype'])

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output


if __name__ == "__main__":
    model_path = "model.tflite"  # Update if stored elsewhere
    model = TFLiteModel(model_path)

    # Example input: adjust shape to match your model
    dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)

    output = model.predict(dummy_input)
    print("Inference output:", output)
