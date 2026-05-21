#!/usr/bin/env python
# Downloads ScanNet public data release
# Run with ./download-scannet.py (or python download-scannet.py on Windows)
# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request
# import urllib
import tempfile

BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean.segs.json', '_vh_clean.aggregation.json', '_vh_clean_2.labels.ply', '_2d-instance.zip', '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']
FILETYPES_TEST = ['.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.ply']
PREPROCESSED_FRAMES_FILE = ['scannet_frames_25k.zip', '5.6GB']
TEST_FRAMES_FILE = ['scannet_frames_test.zip', '610MB']
LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']
DATA_EFFICIENT_FILES = ['limited-reconstruction-scenes.zip', 'limited-annotation-points.zip', 'limited-bboxes.zip', '1.7MB']
GRIT_FILES = ['ScanNet-GRIT.zip']
RELEASES = ['v2/scans', 'v1/scans']
RELEASES_TASKS = ['v2/tasks', 'v1/tasks']
RELEASES_NAMES = ['v2', 'v1']
RELEASE = RELEASES[0]
RELEASE_TASKS = RELEASES_TASKS[0]
RELEASE_NAME = RELEASES_NAMES[0]
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
RELEASE_SIZE = '1.2TB'
V1_IDX = 1


def get_release_scans(release_file):
    """Fetch the list of scan IDs from a remote release manifest file.

    Called early in ``main`` to retrieve the full set of scan identifiers for
    the chosen ScanNet release (v1 or v2), as well as the test-set scans.
    The manifest is a plain-text file hosted on the ScanNet server with one
    scan ID per line.

    Steps:
        1. Open the remote URL specified by ``release_file`` via urllib.
        2. Read each line, decode from bytes to UTF-8, and strip the newline.
        3. Append the cleaned scan ID to the result list.

    Args:
        release_file (str): Full URL pointing to the release manifest text
            file (e.g., ``BASE_URL + 'v2/scans.txt'``).

    Returns:
        list[str]: A list of scan ID strings, e.g. ``['scene0000_00', ...]``.
    """
    scan_lines = urllib.request.urlopen(release_file)
    # scan_lines = urllib.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans


def download_release(release_scans, out_dir, file_types, use_v1_sens):
    """Download all scans in a ScanNet release to a local directory.

    Called by ``main`` when the user requests the full dataset download
    (no ``--id`` flag). It iterates over every scan ID in the release
    manifest and delegates to ``download_scan`` for the actual per-file
    retrieval.

    Steps:
        1. Return immediately if ``release_scans`` is empty.
        2. Print a status message indicating the release version and output
           directory.
        3. For each scan ID, construct its output sub-directory and call
           ``download_scan`` with the requested file types.
        4. Print a completion message.

    Args:
        release_scans (list[str]): List of scan IDs to download.
        out_dir (str): Local root directory where scan sub-directories will
            be created.
        file_types (list[str]): File-type suffixes to download per scan
            (e.g., ``['.sens', '_vh_clean.ply']``).
        use_v1_sens (bool): If True, fetch ``.sens`` files from the v1
            release path instead of v2.

    Returns:
        None
    """
    if len(release_scans) == 0:
        return
    print('Downloading ScanNet ' + RELEASE_NAME + ' release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' release.')


def download_file(url, out_file):
    """Download a single file from a URL to a local path, with atomic rename.

    This is the low-level download helper called by all other download
    functions. It creates the parent directory if needed, skips files that
    already exist, and uses a temporary file plus atomic rename to avoid
    leaving partial downloads on disk.

    Steps:
        1. Derive the parent directory from ``out_file`` and create it if it
           does not exist.
        2. If ``out_file`` already exists, print a warning and skip.
        3. Otherwise, create a temporary file in the same directory via
           ``tempfile.mkstemp``, close the file descriptor, then download
           the URL content into the temporary file with ``urllib.request.urlretrieve``.
        4. Atomically rename the temporary file to ``out_file``.

    Args:
        url (str): The remote URL to download from.
        out_file (str): The local file path where the download will be saved.

    Returns:
        None
    """
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        # urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    """Download all requested file types for a single ScanNet scan.

    Called by ``download_release`` for bulk downloads or directly by ``main``
    when the user specifies ``--id``. For each file type, it constructs the
    remote URL (using the v1 path for ``.sens`` when ``use_v1_sens`` is True)
    and delegates to ``download_file``.

    Steps:
        1. Create the output directory if it does not exist.
        2. Iterate over ``file_types``; for each suffix, determine whether to
           use the v1 or v2 release path for the URL.
        3. Call ``download_file(url, out_file)`` for each file.
        4. Print a completion message for the scan.

    Args:
        scan_id (str): The scan identifier, e.g. ``'scene0000_00'``.
        out_dir (str): Local directory where the scan files will be saved.
        file_types (list[str]): File-type suffixes to download
            (e.g., ``['.sens', '_vh_clean_2.ply']``).
        use_v1_sens (bool): If True, download ``.sens`` files from the v1
            release URL instead of the current release.

    Returns:
        None
    """
    print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == '.sens'
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[V1_IDX] + '/' + scan_id + '/' + scan_id + ft
        out_file = out_dir + '/' + scan_id + ft
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)


def download_task_data(out_dir):
    """Download ScanNet v1 task data including label maps and trained models.

    Called by ``main`` when the ``--task_data`` flag is set. It retrieves the
    v1 label mapping file, object classification data and models, and voxel
    labeling data and models from the ScanNet server.

    Steps:
        1. Build the list of task-related file paths (label map, classification
           data/models, voxel labeling data/models).
        2. For each file, construct the remote URL from the v1 tasks base path.
        3. Create local directories as needed and call ``download_file``.
        4. Print completion status.

    Args:
        out_dir (str): Local root directory for task data downloads.

    Returns:
        None
    """
    print('Downloading ScanNet v1 task data...')
    files = [
        LABEL_MAP_FILES[V1_IDX], 'obj_classification/data.zip',
        'obj_classification/trained_models.zip', 'voxel_labeling/data.zip',
        'voxel_labeling/trained_models.zip'
    ]
    for file in files:
        url = BASE_URL + RELEASES_TASKS[V1_IDX] + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded task data.')

def download_tfrecords(in_dir, out_dir):
    """Download Google TensorFlow records for 3D semantic segmentation/detection.

    Called by ``main`` when the ``--tf_semantic`` flag is set. The records are
    split into high-resolution and low-resolution variants, each sharded
    across train, val, and test splits. Total size is approximately 302 GB.

    Steps:
        1. Create the output root directory if it does not exist.
        2. For each resolution folder (``hires_tfrecords``, ``lores_tfrecords``),
           create the corresponding local directory.
        3. For each split (train/val/test) and shard index, construct the
           remote URL and local path, then call ``download_file``.

    Args:
        in_dir (str): Remote base directory URL containing the tfrecord
            folders (e.g., ``BASE_URL + 'v2/tasks/tf3d'``).
        out_dir (str): Local directory where tfrecord folders and files
            will be saved.

    Returns:
        None
    """
    print('Downloading tf records (302 GB)...')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    split_to_num_shards = {'train': 100, 'val': 25, 'test': 10}

    for folder_name in ['hires_tfrecords', 'lores_tfrecords']:
        folder_dir = '%s/%s' % (in_dir, folder_name)
        save_dir = '%s/%s' % (out_dir, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for split, num_shards in split_to_num_shards.items():
            for i in range(num_shards):
                file_name = '%s-%05d-of-%05d.tfrecords' % (split, i, num_shards)
                url = '%s/%s' % (folder_dir, file_name)
                localpath = '%s/%s/%s' % (out_dir, folder_name, file_name)
                download_file(url, localpath)

def download_label_map(out_dir):
    """Download the ScanNet label mapping TSV file for the active release.

    Called by ``main`` when the ``--label_map`` flag is set, or automatically
    as part of a full v2 dataset download. The label map file maps raw
    ScanNet annotation labels to NYU40 and other label sets.

    Steps:
        1. Print a status message with the release version.
        2. Construct the remote URL from the tasks base path and the global
           ``LABEL_MAP_FILE`` name.
        3. Create the local directory if needed and call ``download_file``.
        4. Print a completion message.

    Args:
        out_dir (str): Local directory where the label map file will be saved.

    Returns:
        None
    """
    print('Downloading ScanNet ' + RELEASE_NAME + ' label mapping file...')
    files = [ LABEL_MAP_FILE ]
    for file in files:
        url = BASE_URL + RELEASE_TASKS + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' label mapping file.')


def main():
    """Parse CLI arguments and orchestrate the ScanNet data download workflow.

    This is the entry point of the script, invoked when run from the command
    line. It supports downloading the full release, individual scans, specific
    file types, task data, label maps, preprocessed frames, 2D test frames,
    data-efficient task files, TensorFlow records, and GRIT files.

    Steps:
        1. Parse command-line arguments (output directory, flags for various
           download modes, optional scan ID and file type filters).
        2. If ``--v1`` is set, switch global release variables to v1 paths.
        3. Fetch the release and test-release scan manifests via
           ``get_release_scans``.
        4. Dispatch to the appropriate download function based on the flags:
           ``--task_data``, ``--label_map``, ``--preprocessed_frames``,
           ``--test_frames_2d``, ``--data_efficient``, ``--tf_semantic``,
           ``--grit``, ``--id``, or a full release download.
        5. For a full release download, also retrieve label maps, test scans,
           test frames, and data-efficient files (v2 only).

    Args:
        None (reads from ``sys.argv`` via ``argparse``).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Downloads ScanNet public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', action='store_true', help='download task data (v1)')
    parser.add_argument('--label_map', action='store_true', help='download label map file')
    parser.add_argument('--v1', action='store_true', help='download ScanNet v1 instead of v2')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--preprocessed_frames', action='store_true', help='download preprocessed subset of ScanNet frames (' + PREPROCESSED_FRAMES_FILE[1] + ')')
    parser.add_argument('--test_frames_2d', action='store_true', help='download 2D test frames (' + TEST_FRAMES_FILE[1] + '; also included with whole dataset download)')
    parser.add_argument('--data_efficient', action='store_true', help='download data efficient task files; also included with whole dataset download)')
    parser.add_argument('--tf_semantic', action='store_true', help='download google tensorflow records for 3D segmentation / detection')
    parser.add_argument('--grit', action='store_true', help='download ScanNet files for General Robust Image Task')
    parser.add_argument('--type', help='specific file type to download (.aggregation.json, .sens, .txt, _vh_clean.ply, _vh_clean_2.0.010000.segs.json, _vh_clean_2.ply, _vh_clean.segs.json, _vh_clean.aggregation.json, _vh_clean_2.labels.ply, _2d-instance.zip, _2d-instance-filt.zip, _2d-label.zip, _2d-label-filt.zip)')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the ScanNet terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = 'aaaa'

    if args.v1:
        global RELEASE
        global RELEASE_TASKS
        global RELEASE_NAME
        global LABEL_MAP_FILE
        RELEASE = RELEASES[V1_IDX]
        RELEASE_TASKS = RELEASES_TASKS[V1_IDX]
        RELEASE_NAME = RELEASES_NAMES[V1_IDX]
        LABEL_MAP_FILE = LABEL_MAP_FILES[V1_IDX]
        assert((not args.tf_semantic) and (not args.grit)), "Task files specified invalid for v1"

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)
    file_types = FILETYPES;
    release_test_file = BASE_URL + RELEASE + '_test.txt'
    release_test_scans = get_release_scans(release_test_file)
    file_types_test = FILETYPES_TEST;
    out_dir_scans = os.path.join(args.out_dir, 'scans')
    out_dir_test_scans = os.path.join(args.out_dir, 'scans_test')
    out_dir_tasks = os.path.join(args.out_dir, 'tasks')

    if args.type:  # download file type
        file_type = args.type
        if file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = [file_type]
        if file_type in FILETYPES_TEST:
            file_types_test = [file_type]
        else:
            file_types_test = []
    if args.task_data:  # download task data
        download_task_data(out_dir_tasks)
    elif args.label_map:  # download label map file
        download_label_map(args.out_dir)
    elif args.preprocessed_frames:  # download preprocessed scannet_frames_25k.zip file
        if args.v1:
            print('ERROR: Preprocessed frames only available for ScanNet v2')
        print('You are downloading the preprocessed subset of frames ' + PREPROCESSED_FRAMES_FILE[0] + ' which requires ' + PREPROCESSED_FRAMES_FILE[1] + ' of space.')
        download_file(os.path.join(BASE_URL, RELEASE_TASKS, PREPROCESSED_FRAMES_FILE[0]), os.path.join(out_dir_tasks, PREPROCESSED_FRAMES_FILE[0]))
    elif args.test_frames_2d:  # download test scannet_frames_test.zip file
        if args.v1:
            print('ERROR: 2D test frames only available for ScanNet v2')
        print('You are downloading the 2D test set ' + TEST_FRAMES_FILE[0] + ' which requires ' + TEST_FRAMES_FILE[1] + ' of space.')
        download_file(os.path.join(BASE_URL, RELEASE_TASKS, TEST_FRAMES_FILE[0]), os.path.join(out_dir_tasks, TEST_FRAMES_FILE[0]))
    elif args.data_efficient: # download data efficient task files
        print('You are downloading the data efficient task files' + ' which requires ' + DATA_EFFICIENT_FILES[-1] + ' of space.')
        for k in range(len(DATA_EFFICIENT_FILES)-1):
            download_file(os.path.join(BASE_URL, RELEASE_TASKS, DATA_EFFICIENT_FILES[k]), os.path.join(out_dir_tasks, DATA_EFFICIENT_FILES[k]))
    elif args.tf_semantic: # download google tf records
        download_tfrecords(os.path.join(BASE_URL, RELEASE_TASKS, 'tf3d'), os.path.join(out_dir_tasks, 'tf3d'))
    elif args.grit: # download GRIT file
        download_file(os.path.join(BASE_URL, RELEASE_TASKS, GRIT_FILES[0]), os.path.join(out_dir_tasks, GRIT_FILES[0]))
    elif args.id:  # download single scan
        scan_id = args.id
        is_test_scan = scan_id in release_test_scans
        if scan_id not in release_scans and (not is_test_scan or args.v1):
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(out_dir_scans, scan_id) if not is_test_scan else os.path.join(out_dir_test_scans, scan_id)
            scan_file_types = file_types if not is_test_scan else file_types_test
            use_v1_sens = not is_test_scan
            # if not is_test_scan and not args.v1 and '.sens' in scan_file_types:
                # print('Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press \'n\' to exclude downloading .sens files for each scan')
                # key = input('')
                # if key.strip().lower() == 'n':
                #     scan_file_types.remove('.sens')
            download_scan(scan_id, out_dir, scan_file_types, use_v1_sens)
    else:  # download entire release
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire ScanNet ' + RELEASE_NAME + ' release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all ScanNet ' + RELEASE_NAME + ' scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        # key = input('')
        # if not args.v1 and '.sens' in file_types:
        #     print('Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press \'n\' to exclude downloading .sens files for each scan')
        #     key = input('')
        #     if key.strip().lower() == 'n':
        #         file_types.remove('.sens')
        download_release(release_scans, out_dir_scans, file_types, use_v1_sens=True)
        if not args.v1:
            download_label_map(args.out_dir)
            download_release(release_test_scans, out_dir_test_scans, file_types_test, use_v1_sens=False)
            download_file(os.path.join(BASE_URL, RELEASE_TASKS, TEST_FRAMES_FILE[0]), os.path.join(out_dir_tasks, TEST_FRAMES_FILE[0]))
            for k in range(len(DATA_EFFICIENT_FILES)-1):
                download_file(os.path.join(BASE_URL, RELEASE_TASKS, DATA_EFFICIENT_FILES[k]), os.path.join(out_dir_tasks, DATA_EFFICIENT_FILES[k]))


if __name__ == "__main__": main()