import os
import warnings

import nibabel as nib
import numpy as np
import scipy.io as sio
import scipy.signal
import scipy.sparse

from util.path import get_root


# TODO include logging (instead of the commented prints)


def load_subjects(list_url):
    with open(list_url, 'r') as f:
        subjects = [s.strip() for s in f.readlines()]

    return subjects


def process_subject(parc, subject, tasks, loaders):
    hcp_downloader = loaders[0]
    git_downloader = loaders[1]

    data = dict()

    # print('Processing subject {}'.format(subject))
    task_list = dict()

    for task in tasks:

        # print('\n--- task {} ...'.format(task))

        task_dict = dict()

        ts = get_ts(subject, task, parc, hcp_downloader)

        cues = get_all_cue_times(subject, task, hcp_downloader)

        cue_list = [cues['lf'], cues['lh'], cues['rf'], cues['rh'], cues['t']]

        # TODO magic number
        cue_arr = np.zeros((len(cue_list), 284), dtype=int)

        for i in range(len(cue_list)):
            limb = cue_list[i]
            for j in limb:
                cue_arr[i, j] = 1

        heart, resp = get_vitals(subject, task, hcp_downloader)

        task_dict['ts'] = ts
        task_dict['cues'] = cue_arr
        task_dict['heart'] = heart
        task_dict['resp'] = resp

        task_list[task] = task_dict

    data['functional'] = task_list

    data['adj'] = get_adj(subject, parc, loaders).tocsr()

    return data


def get_ts(subject, task, parc, settings):
    # get time series
    ts = load_ts_for_subject_task(subject, task, settings)

    # get parcellation
    parc_vector, parc_labels = get_parcellation(parc, subject, settings)

    # parcellate time series
    ts_p = parcellate(ts, parc, parc_vector, parc_labels)

    return ts_p


def load_ts_for_subject_task(subject, task, hcp_downloader):
    # print("  loading time series...", end="", flush=True)

    fname = 'tfMRI_' + task + '_Atlas.dtseries.nii'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)

    path = '/tmp/' + furl
    hcp_downloader.load(path)

    ts = np.array(nib.load(path).get_data())

    return ts


def get_cues(subject, task, hcp_downloader):
    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    path = '/tmp/' + fpath
    files = ['cue.txt', 'lf.txt', 'lh.txt', 'rf.txt', 'rh.txt', 't.txt']

    for file in files:
        new_path = os.path.join(path, file)
        hcp_downloader.load(new_path)

    files = os.listdir(path)
    cues = [file.split('.')[0] for file in files if file != 'Sync.txt']

    return cues


def get_cue_times(cue, subject, task, settings, TR):
    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    fpath = '/tmp/' + fpath
    furl = os.path.join(fpath, cue + '.txt')
    with open(furl) as inp:
        evs = [line.strip().split('\t') for line in inp]
        evs_t = [int(float(evi[0]) / TR) for evi in evs]
    return evs_t


def get_all_cue_times(subject, task, hcp_downloader):
    # print("  reading cue signals...", end="", flush=True)

    TR = 0.72
    cues = {cue: get_cue_times(cue, subject, task, hcp_downloader, TR) for cue in
            get_cues(subject, task, hcp_downloader)}

    # print("done.")

    return cues


def get_parcellation(parc, subject, hcp_downloader):
    # print("  reading parcellation...", end="", flush=True)

    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k')

    suffixes = {'aparc': '.aparc.a2009s.32k_fs_LR.dlabel.nii',
                'dense': '.aparc.a2009s.32k_fs_LR.dlabel.nii'}

    parc_furl = os.path.join('/tmp/', fpath, subject + suffixes[parc])

    hcp_downloader.load(parc_furl)
    parc_obj = nib.load(parc_furl)

    if parc == 'aparc':
        parc_vector = np.array(parc_obj.get_data()[0], dtype='int')
        table = parc_obj.header.matrix[0].named_maps.__next__().label_table
        parc_labels = [(region[1].key, region[1].label) for region in table.items()]

    if parc == 'dense':
        n_regions = len(parc_obj.get_data()[0])
        parc_vector = np.array(range(n_regions))
        parc_labels = [(i, i) for i in range(n_regions)]

    # print("done.")

    return parc_vector, parc_labels


def parcellate(ts, parc, parc_vector, parc_labels):
    # print("  performing parcellation...", end="", flush=True)

    tst = ts[:, :parc_vector.shape[0]]

    parc_idx = list(np.unique(parc_vector))

    bad_regions = [label[0] for label in parc_labels if label[1] == '???']
    for bad_region in bad_regions:
        parc_idx.remove(bad_region)

    if parc == 'aparc':
        # build parcellated signal by taking the mean across voxels
        x_parc = np.array([np.mean(tst[:, (parc_vector == i).tolist()], axis=1) for i in parc_idx])

    if parc == 'dense':
        x_parc = tst.T

    # print("done.")

    return x_parc


def get_vitals(subject, task, hcp_downloader):
    # print("  reading vitals...", end="", flush=True)

    TR = 0.72

    fname = 'tfMRI_' + task + '_Physio_log.txt'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)
    path = '/tmp/' + furl

    hcp_downloader.load(path)

    with open(path) as inp:
        phy = [line.strip().split('\t') for line in inp]
        resp = [int(phy_line[1]) for phy_line in phy]
        heart = [int(phy_line[2]) for phy_line in phy]

    # base TR period fMRI sampling rate

    fl = 1 / TR
    # physio sampling rate
    fh = 400
    # decimation order
    n = 2
    q = int(fh / fl)
    heart_d = scipy.signal.decimate(np.array(heart[:-1]), q, n, zero_phase=True)
    resp_d = scipy.signal.decimate(np.array(resp[:-1]), q, n, zero_phase=True)

    # print("done.")

    return heart_d, resp_d


def read_surf_to_gray_map(hemi):
    fname = os.path.join(get_root(), 'dataset', 'hcp', 'res', hemi + '_dense_map.txt')

    surf_to_gray = np.loadtxt(fname, delimiter=',', dtype=int)

    return surf_to_gray


def map_to_surf(idx, surf_to_gray):
    surf_idx = np.nonzero(surf_to_gray[:, 1] == idx)[0]

    if surf_idx.shape[0] == 0:
        to_idx = -1
    else:
        to_idx = surf_to_gray[int(surf_idx), 0]

    return to_idx


def get_row_cols(faces, hemi):
    rows = list()
    cols = list()

    surf_to_gray = read_surf_to_gray_map(hemi)

    for i in faces:

        p1 = map_to_surf(i[0], surf_to_gray)
        p2 = map_to_surf(i[1], surf_to_gray)
        p3 = map_to_surf(i[2], surf_to_gray)

        if p1 > 0 and p2 > 0:
            rows.append(p1)
            cols.append(p2)
            rows.append(p2)
            cols.append(p1)

        if p1 > 0 and p3 > 0:
            rows.append(p1)
            cols.append(p3)
            rows.append(p3)
            cols.append(p1)

        if p2 > 0 and p3 > 0:
            rows.append(p2)
            cols.append(p3)
            rows.append(p3)
            cols.append(p2)

    return rows, cols


def filter_surf_vertices(coords):
    new_coords = []

    hemi = 'L'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    hemi = 'R'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    new_coords = np.array(new_coords)

    return new_coords


def get_adj_hemi(hemi, inflation, subject, hcp_downloader, offset):
    fname = subject + '.' + hemi + '.' + inflation + '.32k_fs_LR.surf.gii'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k', fname)
    file = '/tmp/' + furl

    hcp_downloader.load(file)
    img = nib.load(file)

    coords = img.darrays[0].data
    faces = img.darrays[1].data.astype(int) + offset
    rows, cols = get_row_cols(faces, hemi)
    new_coords = filter_surf_vertices(coords)

    return rows, cols, new_coords


def get_adj(subject, parc, loaders):
    if parc == 'aparc':
        adj = get_adj_dti(subject, parc, loaders[1])

    elif parc == 'dense':
        adj, coords = get_adj_mesh(subject, loaders[0])

    return adj


def get_adj_dti(subject, parc, git_downloader):
    # print("\n--- dti adjacency matrix...", end="", flush=True)
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'dMRI_CONN')
    file = '/tmp/' + furl + '/' + subject + '.aparc.a2009s.dti.conn.mat'

    git_downloader.load(file)

    try:
        S = sio.loadmat(file)
        S = S.get('S')
    except:
        file_dir = os.path.join(get_root(), 'load/res/average1.aparc.a2009s.dti.conn.mat')
        S = sio.loadmat(file_dir).get('S')
        warnings.warn(
            "Local DTI adjacency matrix for subject: <subject> in parcellation:<parcellation>  not available, using average adjacency matrix.")

    S_coo = scipy.sparse.coo_matrix(S)

    return S_coo


def get_adj_mesh(subject, settings):
    # print("\n--- mesh adjacency matrix...")

    inflation = 'inflated'  # 'white'

    # print("  processing left hemisphere edges...", end="", flush=True)
    hemi = 'L'
    rows_L, cols_L, coords_L = get_adj_hemi(hemi, inflation, subject, settings, offset=0)
    # print("done.")

    # print("  processing right hemisphere edges...", end="", flush=True)
    hemi = 'R'
    rows_R, cols_R, coords_R = get_adj_hemi(hemi, inflation, subject, settings, offset=0)
    # print("done.")

    # print("  processing coordinates...", end="", flush=True)
    data = np.ones(len(rows_L) + len(rows_R))
    A = scipy.sparse.coo_matrix((data, (rows_L + rows_R, cols_L + cols_R)))
    coords = np.vstack((coords_L, coords_R))
    new_coords = filter_surf_vertices(coords)
    # print("done.")

    return A, new_coords


def encode(C, X, H, Gp, Gn):
    """
    encodes
    :param C: data labels
    :param X: data to be windowed
    :param H: window size
    :param Gp: start point guard
    :param Gn: end point guard
    :return:
    """
    _, m, _ = C.shape
    Np, p, T = X.shape
    N = T - H + 1
    num_examples = Np * N

    y = np.zeros([Np, N])
    C_temp = np.zeros(T)

    for i in range(Np):
        for j in range(m):
            temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
            cue_idx1 = [idx - Gn for idx in temp_idx]
            cue_idx2 = [idx + Gp for idx in temp_idx]
            cue_idx = list(zip(cue_idx1, cue_idx2))

            for idx in cue_idx:
                C_temp[slice(*idx)] = j + 1

        y[i, :] = C_temp[0: N]

    X_windowed = np.zeros([Np, N, p, H])

    for t in range(N):
        X_windowed[:, t, :, :] = X[:, :, t: t + H]

    y = np.reshape(y, (num_examples))
    X_windowed = np.reshape(X_windowed, (num_examples, p, H))

    return [X_windowed.astype("float32"), y]
