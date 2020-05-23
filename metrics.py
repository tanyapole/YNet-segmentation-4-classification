from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np
from Utils.constants import YNET, PRETRAIN


def jaccard(y_true, y_pred):
    intersection = np.sum(np.abs(y_true * y_pred))
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred))
    jac = (intersection ) / (sum_ - intersection + 1e-10)
    return jac


class Metric:

    def __init__(self, args):

        self.conf_matrix = np.zeros([len(args.attribute), 2, 2])
        self.true_segm = np.array([])
        self.pred_segm = np.array([])
        self.concat = False
        self.loss = []

    def update(self, y_true, y_pred, loss, train_type:str):

        if train_type == PRETRAIN:
            y_pred = (y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1).data.cpu().numpy() > 0) * 1
            y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1).data.cpu().numpy()
            if not self.concat:
                self.true_segm = y_true
                self.pred_segm = y_pred
                self.concat = True
            else:
                self.true_segm = np.concatenate([self.true_segm, y_true], axis=0)
                self.pred_segm = np.concatenate([self.pred_segm, y_pred], axis=0)
        else:
            y_pred = (y_pred.data.cpu().numpy() > 0) * 1
            y_true = y_true.data.cpu().numpy()
            loss = loss.detach().cpu().numpy().item()

            if y_true.shape[1] == 1:
                self.conf_matrix += confusion_matrix(y_true, y_pred, labels=[0, 1])
            else:
                self.conf_matrix += multilabel_confusion_matrix(y_true, y_pred)

        self.loss.append(loss)

    def compute(self, ep: int, epoch_time: float, train_type: str) -> dict:


        acc_l = []
        prec_l =[]
        rec_l = []
        f1_l  = []


        if train_type == PRETRAIN:
            acc = 0.
            prec = 0.
            rec = 0.
            f1 = 0.
            jac = jaccard(self.true_segm, self.pred_segm)
        else:
            for cm in self.conf_matrix:
                tn, fp, fn, tp = cm.ravel()
                acc_l.append((tp + tn) / (tp + tn + fp + fn))  # TP+TN/(TP+TN+FP+FN)
                p = tp / (tp + fp + 1e-15)  # TP   /(TP+FP)
                prec_l.append(p)
                r = tp / (tp + fn + 1e-15)  # TP   /(TP+FN)
                rec_l.append(r)
                f1_l.append(2 * p * r / (p + r + 1e-15))  # 2*PREC*REC/(PREC+REC)
            acc = sum(acc_l) / len(acc_l)
            prec = sum(prec_l) / len(prec_l)
            rec = sum(rec_l) / len(rec_l)
            f1 = sum(f1_l) / len(f1_l)
            jac = 0

        loss = sum(self.loss) / len(self.loss)

        return {'epoch': ep,
                'loss': loss,
                'epoch_time': epoch_time,
                'accuracy': acc,
                'accuracy_labels': acc_l,
                'precision': prec,
                'precision_labels': prec_l,
                'recall' : rec,
                'recall_labels': rec_l,
                'f1_score': f1,
                'f1_score_labels': f1_l,
                'jaccard': jac
            }

    def reset(self):
        self.conf_matrix = np.zeros([self.conf_matrix.shape[0], 2, 2])
        self.true_segm = np.array([])
        self.pred_segm = np.array([])
        self.concat = False
        self.loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
