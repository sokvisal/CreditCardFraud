from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


def smoteenn(x_train, y_train):
    # oversample minority and undersample majority so that ratio of minority to majority class is 0.3
    smote = SMOTEENN(sampling_strategy=0.3, random_state=42)

    # generate new dataset with syntheic observations
    resample_x_train, resample_y_train = smote.fit_resample(x_train, y_train)

    return resample_x_train, resample_y_train
