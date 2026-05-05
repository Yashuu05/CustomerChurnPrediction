import os
import sys
# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
from imblearn.over_sampling import SMOTE
from src.logger import logging

def perform_smote(Xtrain, Ytrain):
    """
    This function performs SMOTE to increase the number of minority class
    from dataset to avoid overfitting of model
    Input: imbalanced training dataset
    output: balaned training dataset
    """

    print(f"before oversampling, counts of label '1': {sum(Ytrain==1)}")
    print(f"before oversampling, counts of label '0' : {sum(Ytrain==0)}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(Xtrain, Ytrain)

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

    return X_train_res, y_train_res


if __name__ == "__main__":
    
