import os

DATASET_DEFS = {
    'beatles': {
        'data_dir': 'beatles',
        'split_filename': '8-fold_cv_album_distributed_{}.fold'
    },
    'queen': {
        'data_dir': 'queen',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'zweieck': {
        'data_dir': 'zweieck',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'robbie_williams': {
        'data_dir': 'robbie_williams',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'rwc': {
        'data_dir': 'rwc',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'billboard': {
        'data_dir': os.path.join('mcgill-billboard', 'unique'),
        'split_filename': '8-fold_cv_random_{}.fold'
    }
}


def load_dataset(name, data_dir, feature_cache_dir,
                 compute_features, compute_targets):

    data_dir = os.path.join(data_dir, DATASET_DEFS[name]['data_dir'])
    split_filename = os.path.join(data_dir, 'splits', DATASET_DEFS[name]['split_filename'])

    return dmgr.Dataset(
        data_dir,
        os.path.join(feature_cache_dir, name),
        [split_filename.format(f) for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
    )