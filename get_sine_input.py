import numpy as np
import pickle

def generate_sine_inputs(T=20, dt=0.01, f1=1.0, f2=3.5):
    t = np.arange(0, T, dt)
    sin_x = np.sin(1 * np.pi * f1 * t)
    sin_y = np.sin(1 * np.pi * f2 * t)
    inputs = np.vstack([sin_x, sin_y])
    return inputs, dt

if __name__ == "__main__":
    inputs, dt = generate_sine_inputs()
    with open("sines.pkl", "wb") as f:
        pickle.dump({'inputs': inputs, 'dt': dt}, f)
    print(f"Saved sine input with shape {inputs.shape} and dt {dt} to sines.pkl")
