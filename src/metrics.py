"""Raw-point semantic metrics: grouped primary and original 18-class auxiliary."""
import numpy as np


def confusion_matrix(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError('Exactly one prediction per original point is required')
    for values in (target, prediction):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Integer class IDs required; apply 18-class argmax before evaluation')
        if np.any((values < 0) | (values > 17)):
            raise ValueError('Invalid or uncovered point; IDs must be 0..17')
    indices = target.astype(np.int64) * 18 + prediction.astype(np.int64)
    return np.bincount(indices, minlength=324).reshape(18, 18)


def _summary(counts):
    tp = np.diag(counts)
    union = counts.sum(axis=0) + counts.sum(axis=1) - tp
    present = union > 0
    values = np.full(len(tp), np.nan, dtype=np.float64)
    np.divide(tp, union, out=values, where=present)
    return {'iou': [float(x) if np.isfinite(x) else None for x in values],
            'miou': float(values[present].mean()) if present.any() else None,
            'evaluated_classes': int(present.sum()), 'union': union.tolist()}


def evaluate(counts):
    counts = np.asarray(counts)
    if counts.shape != (18, 18) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError('An 18x18 nonnegative integer confusion matrix is required')
    grouped = np.zeros((2, 2), dtype=np.int64)
    mapping = np.zeros(18, dtype=np.int64)
    mapping[[5, 6]] = 1
    for truth in range(18):
        for predicted in range(18):
            grouped[mapping[truth], mapping[predicted]] += counts[truth, predicted]
    primary, auxiliary = _summary(grouped), _summary(counts)
    valid_primary = primary['evaluated_classes'] == 2
    # An absent group must never silently turn the two-group criterion into a one-group score.
    if not valid_primary:
        primary['miou'] = None
    return {'points': int(counts.sum()), 'confusion18': counts.tolist(),
            'primary': {'name': 'wall_drywall_vs_other_miou', 'group_order': ['Other', 'Wall+Drywall'],
                        'confusion': grouped.tolist(), 'valid_two_group_score': valid_primary,
                        **primary},
            'auxiliary': {'class_ids': list(range(18)), 'class_iou18': auxiliary['iou'],
                          'miou18': auxiliary['miou'], 'evaluated_classes': auxiliary['evaluated_classes']},
            'scope': 'Counts must be pooled across the complete declared split with original-point coverage; synthetic/pilot scores are not final acceptance'}
