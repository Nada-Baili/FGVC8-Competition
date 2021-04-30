import commentjson, collections, random
from sklearn.metrics import fbeta_score

with open("./params.json") as json_file:
    params = commentjson.load(json_file)

def make_folds(df, n_folds, seed):
    cls_counts = collections.Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)
    fold_cls_counts = collections.defaultdict(int)
    folds = [-1] * len(df)
    for item in df.sample(frac=1, random_state=seed).itertuples():
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df

def get_score(targets, y_pred):
    return fbeta_score(targets, y_pred, beta=2, average='samples')

def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=5):
    """
    Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == params["nb_classes"]
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask

def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask

def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]