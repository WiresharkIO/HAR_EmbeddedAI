import numpy as np
import pandas as pd
from scipy import stats
import os
from tqdm import tqdm

from definitions import ROOT_DIR

sourceFolder = ROOT_DIR

# Input data paths (where the preprocessed data is stored)
dataFolder_inlab = sourceFolder + '/features/sequence/input/inlab_dataset/'
dataFolder_freeliving = sourceFolder + '/features/sequence/input/freeliving_dataset/'
dataFolder_combined = sourceFolder + '/features/sequence/input/combined_dataset/'

# Updated output feature paths to match your directory structure
# featuresFolder_inlab = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/inlab_feature_extraction/'
# featuresFolder_freeliving = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/freeliving_feature_extraction/'
# featuresFolder_combined = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/combined_feature_extraction/'

featuresFolder_inlab = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/64_feature_extraction/inlab_feature_extraction/'
featuresFolder_freeliving = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/64_feature_extraction/freeliving_feature_extraction/'
featuresFolder_combined = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/64_feature_extraction/combined_feature_extraction/'


# Create output directories if they don't exist
os.makedirs(featuresFolder_inlab, exist_ok=True)
os.makedirs(featuresFolder_freeliving, exist_ok=True)
os.makedirs(featuresFolder_combined, exist_ok=True)

participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

Fs = 50

#  128 buffer size
# segment_length = 2.56

# 64 buffer size
segment_length = 1.28

overlap = 0.5
frame_annotation_threshold = 0.3
frame_length = int(segment_length * Fs)
iter_length = frame_length * (1 - overlap)


def load_data_from_file(participant_id, data_type):
    """
    Load data from preprocessed file based on data type
    """
    if data_type == 'inlab':
        file_path = os.path.join(dataFolder_inlab, f'participant_data_{participant_id}_inlab.csv')
    elif data_type == 'freeliving':
        file_path = os.path.join(dataFolder_freeliving, f'participant_data_{participant_id}_freeliving.csv')
    elif data_type == 'combined':
        file_path = os.path.join(dataFolder_combined, f'participant_data_{participant_id}_combined.csv')
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist!")
        return None, None, None

    # Load the data
    df = pd.read_csv(file_path)

    # Extract the columns
    prox_data = df['prox_data'].values
    sequence_annotation = df['sequence_annotation'].values
    eating_annotation = df['eating_annotation'].values

    # Convert prox_data to uint16 if possible
    if prox_data.min() >= 0 and prox_data.max() <= 65535:
        prox_data = prox_data.astype(np.uint16)

    return prox_data, sequence_annotation, eating_annotation


def ema_filter(frame, beta=0.55):
    # Explicitly convert to float32
    frame_float = frame.astype(np.float32)
    filtered = np.zeros(len(frame), dtype=np.float32)
    filtered[0] = frame_float[0]

    for i in range(1, len(frame)):
        filtered[i] = filtered[i - 1] - beta * (filtered[i - 1] - frame_float[i])

    return filtered


def standardize_frame(filtered_frame):
    mean = np.mean(filtered_frame, dtype=np.float32)
    std = np.std(filtered_frame, dtype=np.float32)
    if std != 0:
        # Explicitly ensure float32 output
        return ((filtered_frame - mean) / std).astype(np.float32)
    else:
        return (filtered_frame - mean).astype(np.float32)


def analyze_crossing_patterns(standardized_frame):
    crossings = []
    crossing_amplitudes = []

    for i in range(1, len(standardized_frame)):
        if (standardized_frame[i - 1] * standardized_frame[i]) < 0:
            crossings.append(i)
            # Calculate absolute amplitude change at crossing
            amplitude = abs(standardized_frame[i] - standardized_frame[i - 1])
            crossing_amplitudes.append(amplitude)

    if len(crossings) > 1:
        intervals = np.diff(crossings)
        crossing_amplitudes = np.array(crossing_amplitudes)

        # Population-based interval thresholds
        chew_lower = 13.1 - 8.64  # ≈ 4.5 samples
        chew_upper = 41
        non_chew_upper = 2.89 + 3.96  # ≈ 6.85 samples

        # Classify intervals
        chewing_intervals = (intervals >= chew_lower) & (intervals <= chew_upper)
        non_chewing_intervals = intervals <= non_chew_upper

        # Get amplitudes for each type of crossing
        chewing_amplitudes = crossing_amplitudes[:-1][chewing_intervals]
        non_chewing_amplitudes = crossing_amplitudes[:-1][non_chewing_intervals]

        # Calculate amplitude statistics
        features = {
            'total_crossings': len(crossings),
            'mean_interval': np.mean(intervals),

            # Basic amplitude features
            'mean_amplitude': np.mean(crossing_amplitudes),
            'max_amplitude': np.max(crossing_amplitudes),
            'min_amplitude': np.min(crossing_amplitudes),
            'std_amplitude': np.std(crossing_amplitudes),

            # Amplitude features by type
            'chewing_mean_amp': np.mean(chewing_amplitudes) if len(chewing_amplitudes) > 0 else 0,
            'non_chewing_mean_amp': np.mean(non_chewing_amplitudes) if len(non_chewing_amplitudes) > 0 else 0,
            'chewing_std_amp': np.std(chewing_amplitudes) if len(chewing_amplitudes) > 0 else 0,
            'non_chewing_std_amp': np.std(non_chewing_amplitudes) if len(non_chewing_amplitudes) > 0 else 0,

            # Interval counts and ratios
            'chewing_intervals': np.sum(chewing_intervals),
            'non_chewing_intervals': np.sum(non_chewing_intervals),
            'chewing_ratio': np.sum(chewing_intervals) / len(intervals),
            'non_chewing_ratio': np.sum(non_chewing_intervals) / len(intervals),
        }

        return features

    return {
        'total_crossings': 0,
        'mean_interval': 0,
        'mean_amplitude': 0,
        'max_amplitude': 0,
        'min_amplitude': 0,
        'std_amplitude': 0,
        'chewing_mean_amp': 0,
        'non_chewing_mean_amp': 0,
        'chewing_std_amp': 0,
        'non_chewing_std_amp': 0,
        'chewing_intervals': 0,
        'non_chewing_intervals': 0,
        'chewing_ratio': 0,
        'non_chewing_ratio': 0

    }


def process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation, output_folder):
    resultFile = os.path.join(output_folder, f'sequence_features_{participant_id}.csv')

    dataLength = len(prox_data)
    print('The data length is ' + str(dataLength) + ': 0-' + str(dataLength - 1))
    frame_index = int(dataLength // iter_length) - 1
    print('The frame index is ' + str(frame_index))

    # Initialize feature arrays
    signal_frame = np.zeros((frame_index, frame_length))

    # Crossing interval feature arrays
    ZC_TOTAL = np.zeros(frame_index)
    ZC_MEAN_INTERVAL = np.zeros(frame_index)
    ZC_CHEW_INTERVALS = np.zeros(frame_index)
    ZC_NONCHEW_INTERVALS = np.zeros(frame_index)
    ZC_CHEW_RATIO = np.zeros(frame_index)
    ZC_NONCHEW_RATIO = np.zeros(frame_index)

    # Amplitude feature arrays
    ZC_MEAN_AMP = np.zeros(frame_index)
    ZC_MAX_AMP = np.zeros(frame_index)
    ZC_MIN_AMP = np.zeros(frame_index)
    ZC_STD_AMP = np.zeros(frame_index)
    ZC_CHEW_MEAN_AMP = np.zeros(frame_index)
    ZC_NONCHEW_MEAN_AMP = np.zeros(frame_index)
    ZC_CHEW_STD_AMP = np.zeros(frame_index)
    ZC_NONCHEW_STD_AMP = np.zeros(frame_index)

    # Time-Domain Features
    TD_MAX = np.zeros(frame_index)
    TD_MIN = np.zeros(frame_index)
    TD_MAX_MIN = np.zeros(frame_index)
    TD_RMS = np.zeros(frame_index)
    TD_MEDIAN = np.zeros(frame_index)
    TD_VARIANCE = np.zeros(frame_index)
    TD_STD = np.zeros(frame_index)
    TD_SKEW = np.zeros(frame_index)
    TD_KURT = np.zeros(frame_index)
    TD_IQR = np.zeros(frame_index)

    # area under the curve
    AREA_TOTAL = np.zeros(frame_index)
    AREA_POSITIVE = np.zeros(frame_index)
    AREA_NEGATIVE = np.zeros(frame_index)
    AREA_RATIO = np.zeros(frame_index)

    # Ground Truth
    GROUND_TRUTH_FRAME = np.zeros(frame_index)
    EATING_TRUTH_FRAME = np.zeros(frame_index)
    SEQUENCE_TRUTH_FRAME = np.zeros(frame_index)
    BEGIN_INDEX_FRAME = np.zeros(frame_index)
    END_INDEX_FRAME = np.zeros(frame_index)

    # Process frames
    for i in tqdm(range(0, frame_index), desc="Processing frames"):
        low_count = int(i * iter_length)
        high_count = int(low_count + frame_length)

        # Get and filter frame
        frame = prox_data[low_count:high_count]
        filtered_frame = ema_filter(frame, beta=0.55)
        standardized_frame = standardize_frame(filtered_frame)

        zc_features = analyze_crossing_patterns(standardized_frame)
        # Store interval features
        ZC_TOTAL[i] = zc_features['total_crossings']
        ZC_MEAN_INTERVAL[i] = zc_features['mean_interval']
        ZC_CHEW_INTERVALS[i] = zc_features['chewing_intervals']
        ZC_NONCHEW_INTERVALS[i] = zc_features['non_chewing_intervals']
        ZC_CHEW_RATIO[i] = zc_features['chewing_ratio']
        ZC_NONCHEW_RATIO[i] = zc_features['non_chewing_ratio']

        # Store amplitude features
        ZC_MEAN_AMP[i] = zc_features['mean_amplitude']
        ZC_MAX_AMP[i] = zc_features['max_amplitude']
        ZC_MIN_AMP[i] = zc_features['min_amplitude']
        ZC_STD_AMP[i] = zc_features['std_amplitude']
        ZC_CHEW_MEAN_AMP[i] = zc_features['chewing_mean_amp']
        ZC_NONCHEW_MEAN_AMP[i] = zc_features['non_chewing_mean_amp']
        ZC_CHEW_STD_AMP[i] = zc_features['chewing_std_amp']
        ZC_NONCHEW_STD_AMP[i] = zc_features['non_chewing_std_amp']

        # change to filtered_frame
        TD_MAX[i] = np.max(standardized_frame)
        TD_MIN[i] = np.min(standardized_frame)
        TD_MAX_MIN[i] = TD_MAX[i] - TD_MIN[i]
        TD_RMS[i] = np.sqrt(np.mean(standardized_frame ** 2))
        TD_MEDIAN[i] = np.median(standardized_frame)
        TD_VARIANCE[i] = np.var(standardized_frame)  # Should be close to 1
        TD_STD[i] = np.std(standardized_frame)  # Should be close to 1
        TD_SKEW[i] = stats.skew(standardized_frame)
        TD_KURT[i] = stats.kurtosis(standardized_frame)
        TD_IQR[i] = stats.iqr(standardized_frame)

        # Area under the curve
        AREA_TOTAL[i] = np.trapezoid(np.abs(standardized_frame))

        # Positive area (integral of only positive values)
        positive_only = np.where(standardized_frame > 0, standardized_frame, 0)
        AREA_POSITIVE[i] = np.trapezoid(positive_only)

        # Negative area (integral of absolute negative values)
        negative_only = np.where(standardized_frame < 0, np.abs(standardized_frame), 0)
        AREA_NEGATIVE[i] = np.trapezoid(negative_only)

        # Ratio of positive to total area
        AREA_RATIO[i] = AREA_POSITIVE[i] / AREA_TOTAL[i] if AREA_TOTAL[i] > 0 else 0.5

        # Ground Truth
        if sum(sequence_annotation[low_count:high_count]) > (frame_length * frame_annotation_threshold):
            SEQUENCE_TRUTH_FRAME[i] = 1
        else:
            SEQUENCE_TRUTH_FRAME[i] = 0

        if sum(eating_annotation[low_count:high_count]) > (frame_length * frame_annotation_threshold):
            EATING_TRUTH_FRAME[i] = 1
            GROUND_TRUTH_FRAME[i] = 1
        else:
            EATING_TRUTH_FRAME[i] = 0
            GROUND_TRUTH_FRAME[i] = 0

        BEGIN_INDEX_FRAME[i] = low_count
        END_INDEX_FRAME[i] = high_count

    feature_data = pd.DataFrame()
    feature_data['ZC_TOTAL'] = ZC_TOTAL

    feature_data['ZC_MEAN_INTERVAL'] = ZC_MEAN_INTERVAL
    feature_data['ZC_CHEW_INTERVALS'] = ZC_CHEW_INTERVALS
    feature_data['ZC_NONCHEW_INTERVALS'] = ZC_NONCHEW_INTERVALS
    feature_data['ZC_CHEW_RATIO'] = ZC_CHEW_RATIO
    feature_data['ZC_NONCHEW_RATIO'] = ZC_NONCHEW_RATIO

    feature_data['ZC_MEAN_AMP'] = ZC_MEAN_AMP
    feature_data['ZC_MAX_AMP'] = ZC_MAX_AMP
    feature_data['ZC_MIN_AMP'] = ZC_MIN_AMP
    feature_data['ZC_STD_AMP'] = ZC_STD_AMP
    feature_data['ZC_CHEW_MEAN_AMP'] = ZC_CHEW_MEAN_AMP
    feature_data['ZC_NONCHEW_MEAN_AMP'] = ZC_NONCHEW_MEAN_AMP
    feature_data['ZC_CHEW_STD_AMP'] = ZC_CHEW_STD_AMP
    feature_data['ZC_NONCHEW_STD_AMP'] = ZC_NONCHEW_STD_AMP

    feature_data['TD_MAX'] = TD_MAX
    feature_data['TD_MIN'] = TD_MIN
    feature_data['TD_MAX_MIN'] = TD_MAX_MIN
    feature_data['TD_RMS'] = TD_RMS
    feature_data['TD_MEDIAN'] = TD_MEDIAN
    feature_data['TD_VARIANCE'] = TD_VARIANCE
    feature_data['TD_STD'] = TD_STD
    feature_data['TD_SKEW'] = TD_SKEW
    feature_data['TD_KURT'] = TD_KURT
    feature_data['TD_IQR'] = TD_IQR

    feature_data['AREA_TOTAL'] = AREA_TOTAL
    feature_data['AREA_POSITIVE'] = AREA_POSITIVE
    feature_data['AREA_NEGATIVE'] = AREA_NEGATIVE
    feature_data['AREA_RATIO'] = AREA_RATIO

    feature_data['GROUND_TRUTH_FRAME'] = GROUND_TRUTH_FRAME
    feature_data['EATING_TRUTH_FRAME'] = EATING_TRUTH_FRAME
    feature_data['SEQUENCE_TRUTH_FRAME'] = SEQUENCE_TRUTH_FRAME
    feature_data['BEGIN_INDEX_FRAME'] = BEGIN_INDEX_FRAME
    feature_data['END_INDEX_FRAME'] = END_INDEX_FRAME

    feature_data.fillna(0)

    feature_data.to_csv(resultFile)
    print('Result saved to: ' + resultFile)


def main():
    # Select which dataset type to process
    # Options: 'inlab', 'freeliving', 'combined', 'all'
    dataset_type = 'all'  # Change this to control which dataset is processed

    # Process each participant
    for participant_id in participants:
        print(f"\nProcessing participant {participant_id}")

        if dataset_type == 'inlab' or dataset_type == 'all':
            print(f"Processing inlab data for participant {participant_id}")
            prox_data, sequence_annotation, eating_annotation = load_data_from_file(participant_id, 'inlab')
            if prox_data is not None:
                process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation,
                                           featuresFolder_inlab)
                print(f"Inlab data processed successfully for participant {participant_id}")
            else:
                print(f"No inlab data available for participant {participant_id}")

        if dataset_type == 'freeliving' or dataset_type == 'all':
            print(f"Processing freeliving data for participant {participant_id}")
            prox_data, sequence_annotation, eating_annotation = load_data_from_file(participant_id, 'freeliving')
            if prox_data is not None:
                process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation,
                                           featuresFolder_freeliving)
                print(f"Freeliving data processed successfully for participant {participant_id}")
            else:
                print(f"No freeliving data available for participant {participant_id}")

        if dataset_type == 'combined' or dataset_type == 'all':
            print(f"Processing combined data for participant {participant_id}")
            prox_data, sequence_annotation, eating_annotation = load_data_from_file(participant_id, 'combined')
            if prox_data is not None:
                process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation,
                                           featuresFolder_combined)
                print(f"Combined data processed successfully for participant {participant_id}")
            else:
                print(f"No combined data available for participant {participant_id}")


if __name__ == "__main__":
    main()