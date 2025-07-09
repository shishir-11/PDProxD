from sklearn.svm import SVC as svc
import numpy as np
class SVC():
    def __init__(self, data=None, target=None,
                 C=1,
                 iter=100):
        self.model = svc(C=C,kernel='linear',max_iter=iter,decision_function_shape='ovr')
        self.data = data
        self.target = target
        
    def train(self):
        self.model.fit(self.data,self.target)
    
    def predict(self, X):
        return self.model.predict(X)

    def support_vector_ratio(self):
        return len(self.model.support_) / len(self.data)