from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

class LoadMetric:
    def __init__(self,type='accuracy',average='weighted'):
        self.type = type
        self.average = average
        self.metric = None
    
    def load_metric(self):
        met_dict = {
            "accuracy": accuracy_score,
            "precision":precision_score,
            "recall": recall_score,
            "f1": f1_score
        }

        self.metric = met_dict[self.type]

    def get_score(self,y_test,y_pred):
        if(self.type=='accuracy'):
            return self.metric(y_test,y_pred)    
        return self.metric(y_test,y_pred,average=self.average)
    