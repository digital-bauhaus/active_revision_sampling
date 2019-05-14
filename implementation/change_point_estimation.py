import numpy as np
import scipy.signal as sps

def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm

    return ta, tai, taf

def fuzz_list(vector, maximum, fuzzy=2):
    """
    Matching of CUSUM-detected change points against an estimation obtained
    from GP regression models.

    :param vector: signal, e.g. estimation or ground truth
    :param vector:
    """
    fuzz = []
    for v in vector:
        fuzz.append(v)
        for i in range(1, fuzzy + 1):
            fuzz.append(v - i)
            fuzz.append(v + i)
    fuzz = list(filter(lambda x: x >= 0 and x < maximum, fuzz))
    fuzz = list(set(fuzz))
    return fuzz

def change_point_estimation(ys: np.array, estimation_matrix: np.array, cusum_level: float, training_level: float):
    """
    Matching of CUSUM-detected change points against an estimation obtained
    from GP regression models.

    :param ys: ground truth observation for a single time-series
    :param estimation_matrix: estimation: matrix of shape (a, b), where a is the number of iterations, b the number of estimated revisions
    :param cusum_level: sliding window size as percentage (between 0 and 1), we used (0.01, 0.05, 0.025, 0.1)
    :param training_level: percentage (between 0 and 1) of revisions seen by the learner, we used (0.01, 0.03, 0.05)

    :return: precision and recall for the estimation
    """
    # Calculate the parameters for CUSUM analysis
    drift = 0.05 * (np.max(np.abs(ys)) - np.min(np.abs(ys)))
    sliding_window_size = int(cusum_level * ys.shape[0])
    threshold_h = 5 * np.mean(ys.rolling(window=sliding_window_size, center=True).std())

    # Calculate change point locations
    cps = detect_cusum(ys, drift=drift, threshold=threshold_h)[0]

    # which iteration of the estimation to assess
    iteration_number = ys.shape[0] * training_level
    estimation = estimation_matrix[iteration_number]

    # Obtain the change window
    estimated_changes = np.diff(estimation, n=10)

    # Obtain change point locations from the
    kernel_change_points = sps.find_peaks(estimated_changes, threshold=np.std(ys))[0]
    fuzz_kernel_change_points = list(map(lambda x: fuzz_list([x], fuzzy=5, maximum=len(ys)), kernel_change_points))

    # counters
    true_pos_regions = []
    false_pos_regions = []
    false_negatives = 0

    for region in fuzz_kernel_change_points:
        for c in cps:
            if c in region:
                true_pos_regions.append(region)
                false_pos_regions.append(region)

    for c in cps:
        matched = []
        for region in fuzz_kernel_change_points:
            matched.append(c in region)
        if not any(matched):
            false_negatives += 1

    cleaned_fps = []
    for f_region in false_pos_regions:
        for p_region in true_pos_regions:

            # If the two regions overlap, keep f_region
            if len(set(f_region).intersection(set(p_region))) > 0:
                cleaned_fps.append(f_region)

    true_positives = len(true_pos_regions)
    false_positives = len(cleaned_fps)

    # Calculate precision and recall
    # If the CUSUM analysis did not find anything, set precision/recall to 1
    if len(cps) == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall
