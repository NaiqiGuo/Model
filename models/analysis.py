import numpy as np

class ModalAnalysis:
    def __init__(self, model, n):
        self.model = model
        self.n = n
    
        self.lambdas = model.eigen(n)  
        print(f"[Debug] eigen(n={n}) returned {len(lambdas)} values: {lambdas}")
        lambdas = np.asarray(lambdas, dtype=float)
        omega = np.sqrt(np.abs(lambdas))                    # rad/s
        self.frequencies = omega / (2*np.pi)