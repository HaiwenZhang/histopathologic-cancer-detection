import torch

class BCA:
    def __init__(self, threshold=0.5):
        self.predicts = []
        self.targets = []
        self.threshold = threshold

    def __call__(self, predict, target):
        """
        predict and target are in batch
        """

        self.predicts.append(predict)
        self.targets.append(target)

    def res(self):
        origin_predict = torch.cat(self.predicts)
        target = torch.cat(self.targets)
        predict = origin_predict > self.threshold
        truth = target >= self.threshold
        
        a = predict.eq(truth).sum().cpu().numpy()
        acc = a / target.numel()
        self.predicts = []
        self.targets = []
        return acc

