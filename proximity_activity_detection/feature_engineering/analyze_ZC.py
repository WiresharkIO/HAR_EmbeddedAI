# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from definitions import ROOT_DIR
# # Path setup (assuming these are defined in your project)
# sourceFolder = ROOT_DIR
# dataFolder_inlab_L = sourceFolder + '/datasets/inlab/left/'
# dataFolder_inlab_R = sourceFolder + '/datasets/inlab/right/'
# dataFolder_freeliving_L = sourceFolder + '/datasets/freeliving/'
#
# participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
#
#
# def load_participant_data(participant_id, inlab_left_folder, inlab_right_folder, freeliving_folder):
#     participant_data = {
#         'inlab_left': None,
#         'inlab_right': None,
#         'freeliving_left': None
#     }
#
#     inlab_left = os.path.join(inlab_left_folder, f'participant_data_{participant_id}_left.csv')
#     if os.path.exists(inlab_left):
#         df_inlab_left = pd.read_csv(inlab_left)
#         participant_data['inlab_left'] = {
#             'prox_data': df_inlab_left['L_prox_sen_data'].values,
#             'sequence_annotation': np.where(df_inlab_left['L_chew_cycle_annotation'].values > 0, 1, 0),
#             'eating_annotation': [1 if x == 10000 else x for x in df_inlab_left['Eating_event_truth'].values]
#         }
#
#     inlab_right = os.path.join(inlab_right_folder, f'participant_data_{participant_id}_right.csv')
#     if os.path.exists(inlab_right):
#         df_inlab_right = pd.read_csv(inlab_right)
#         participant_data['inlab_right'] = {
#             'prox_data': df_inlab_right['R_prox_sen_data'].values,
#             'sequence_annotation': np.where(df_inlab_right['R_chew_cycle_annotation'].values > 0, 1, 0),
#             'eating_annotation': [1 if x == 10000 else x for x in df_inlab_right['Eating_event_truth'].values]
#         }
#
#     freeliving_left = os.path.join(freeliving_folder, f'participant_data_{participant_id}_left.csv')
#     if os.path.exists(freeliving_left):
#         df_freeliving_left = pd.read_csv(freeliving_left)
#         participant_data['freeliving_left'] = {
#             'prox_data': df_freeliving_left['L_prox_sen_data'].values,
#             'sequence_annotation': df_freeliving_left['Annotation_Sequence_A'].values,
#             'eating_annotation': [1 if x == 10000 else x for x in df_freeliving_left['Eating_event_truth'].values]
#         }
#
#     return participant_data
#
#
# def combine_participant_sources(data):
#     all_prox_data = []
#     all_sequence_annotations = []
#     all_eating_annotations = []
#
#     for source, source_data in data.items():
#         if source_data is not None:
#             all_prox_data.extend(source_data['prox_data'])
#             all_sequence_annotations.extend(source_data['sequence_annotation'])
#             all_eating_annotations.extend(source_data['eating_annotation'])
#
#     prox_data = np.array(all_prox_data)
#     sequence_annotation = np.array(all_sequence_annotations)
#     eating_annotation = np.array(all_eating_annotations)
#
#     return prox_data, sequence_annotation, eating_annotation
#
#
# def apply_ema_filter(data, alpha=0.55):
#     """Apply Exponential Moving Average filter to the data"""
#     filtered = np.zeros_like(data)
#     filtered[0] = data[0]
#     for i in range(1, len(data)):
#         filtered[i] = filtered[i - 1] - (alpha * (filtered[i - 1] - data[i]))
#     return filtered
#
#
# def standardize_frame(frame):
#     """Standardize a frame by subtracting mean and dividing by standard deviation"""
#     mean = np.mean(frame)
#     std = np.std(frame)
#     if std != 0:
#         return (frame - mean) / std
#     else:
#         return frame - mean  # Just center if std is zero
#
#
# def compute_zero_crossing_intervals(standardized_frame, amplitude_threshold=0.3):
#     """Compute intervals between zero crossings that exceed the amplitude threshold"""
#     crossings = []
#
#     # Find zero crossings with significant amplitude change
#     for i in range(1, len(standardized_frame)):
#         if (standardized_frame[i - 1] * standardized_frame[i]) <= 0:  # Zero crossing
#             amplitude_diff = abs(standardized_frame[i] - standardized_frame[i - 1])
#             if amplitude_diff > amplitude_threshold:  # Apply threshold to filter noise
#                 crossings.append(i)
#
#     # Calculate intervals between consecutive crossings
#     if len(crossings) >= 2:
#         intervals = np.diff(crossings)
#         return intervals
#     else:
#         return np.array([])  # Return empty array if fewer than 2 crossings
#
#
# def analyze_participant_frames(signal, labels, frame_size=128, overlap=64):
#     """Analyze all frames from a participant's data and collect zero crossing intervals by class"""
#     class0_intervals = []
#     class1_intervals = []
#
#     # Process frames with overlap
#     for i in range(0, len(signal) - frame_size, overlap):
#         frame = signal[i:i + frame_size]
#         frame_labels = labels[i:i + frame_size]
#
#         # Determine frame class (0 or 1) based on majority
#         frame_class = 1 if np.mean(frame_labels) > 0.5 else 0
#
#         # Apply EMA filter to frame
#         filtered_frame = apply_ema_filter(frame)
#
#         # Standardize the filtered frame
#         standardized_frame = standardize_frame(filtered_frame)
#
#         # Calculate zero crossing intervals
#         intervals = compute_zero_crossing_intervals(standardized_frame)
#
#         # Add intervals to corresponding class
#         if intervals.size > 0:  # Only if we found intervals
#             if frame_class == 0:
#                 class0_intervals.extend(intervals)
#             else:
#                 class1_intervals.extend(intervals)
#
#     return np.array(class0_intervals), np.array(class1_intervals)
#
#
# # Store results for all participants
# results = []
#
# # Process each participant
# for participant_id in tqdm(participants, desc="Processing participants"):
#     print(f"\nProcessing participant {participant_id}")
#
#     # Load participant data
#     participant_data = load_participant_data(
#         participant_id,
#         dataFolder_inlab_L,
#         dataFolder_inlab_R,
#         dataFolder_freeliving_L
#     )
#
#     # Combine data from all sources
#     prox_data, sequence_annotation, _ = combine_participant_sources(participant_data)
#
#     # Skip if no data available
#     if len(prox_data) == 0:
#         print(f"No data available for participant {participant_id}, skipping.")
#         continue
#
#     # Analyze frames to get intervals for each class
#     class0_intervals, class1_intervals = analyze_participant_frames(prox_data, sequence_annotation)
#
#     # Calculate statistics if intervals were found
#     if class0_intervals.size > 0:
#         class0_mean = np.mean(class0_intervals)
#         class0_min = np.min(class0_intervals)
#         class0_max = np.max(class0_intervals)
#     else:
#         class0_mean = class0_min = class0_max = np.nan
#
#     if class1_intervals.size > 0:
#         class1_mean = np.mean(class1_intervals)
#         class1_min = np.min(class1_intervals)
#         class1_max = np.max(class1_intervals)
#     else:
#         class1_mean = class1_min = class1_max = np.nan
#
#     # Store results
#     results.append({
#         'participant_id': participant_id,
#         'class0_mean_interval': class0_mean,
#         'class0_min_interval': class0_min,
#         'class0_max_interval': class0_max,
#         'class0_interval_count': len(class0_intervals),
#         'class1_mean_interval': class1_mean,
#         'class1_min_interval': class1_min,
#         'class1_max_interval': class1_max,
#         'class1_interval_count': len(class1_intervals)
#     })
#
#     print(
#         f"Class 0 intervals: {len(class0_intervals)}, Mean: {class0_mean:.2f}, Min: {class0_min:.2f}, Max: {class0_max:.2f}")
#     print(
#         f"Class 1 intervals: {len(class1_intervals)}, Mean: {class1_mean:.2f}, Min: {class1_min:.2f}, Max: {class1_max:.2f}")
#
# # Convert results to DataFrame
# results_df = pd.DataFrame(results)
# print("\nResults summary:")
# print(results_df)
#
# # Save results to CSV
# results_df.to_csv('zero_crossing_intervals_by_participant.csv', index=False)
# print("Results saved to 'zero_crossing_intervals_by_participant.csv'")
#
# # Create scatter plot of mean intervals
# plt.figure(figsize=(12, 8))
#
# # Plot non-chewing (class 0) intervals
# plt.scatter(
#     results_df['participant_id'],
#     results_df['class0_mean_interval'],
#     label='Non-Chewing (Class 0)',
#     color='blue',
#     marker='o',
#     s=100,
#     alpha=0.7
# )
#
# # Plot chewing (class 1) intervals
# plt.scatter(
#     results_df['participant_id'],
#     results_df['class1_mean_interval'],
#     label='Chewing (Class 1)',
#     color='red',
#     marker='x',
#     s=100
# )
#
# # Add error bars showing min and max values
# for i, row in results_df.iterrows():
#     # Class 0 error bars
#     if not np.isnan(row['class0_mean_interval']):
#         plt.plot(
#             [row['participant_id'], row['participant_id']],
#             [row['class0_min_interval'], row['class0_max_interval']],
#             color='blue',
#             alpha=0.3
#         )
#
#     # Class 1 error bars
#     if not np.isnan(row['class1_mean_interval']):
#         plt.plot(
#             [row['participant_id'], row['participant_id']],
#             [row['class1_min_interval'], row['class1_max_interval']],
#             color='red',
#             alpha=0.3
#         )
#
# # Add horizontal line showing typical chewing range (15-40)
# plt.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Typical Chewing Range (15-40)')
# plt.axhline(y=40, color='green', linestyle='--', alpha=0.5)
#
# # Add labels and title
# plt.xlabel('Participant ID')
# plt.ylabel('Zero Crossing Interval')
# plt.title('Mean Zero Crossing Intervals by Participant and Class')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Add a text box with summary statistics
# textstr = "\n".join([
#     "Overall Statistics:",
#     f"Class 0 Mean: {results_df['class0_mean_interval'].mean():.2f}",
#     f"Class 1 Mean: {results_df['class1_mean_interval'].mean():.2f}"
# ])
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
#          verticalalignment='top', bbox=props)
#
# plt.tight_layout()
# plt.savefig('zero_crossing_intervals_by_participant.png', dpi=300)
# plt.show()
#
# # Create boxplot to compare distributions
# plt.figure(figsize=(14, 8))
#
# # Prepare data for boxplot
# class0_data = []
# class1_data = []
# ids = []
#
# for _, row in results_df.iterrows():
#     ids.append(row['participant_id'])
#
# # Create boxplot data structure
# boxplot_data = [
#                    {'Participant': pid, 'Interval': val, 'Class': 'Non-Chewing (0)'}
#                    for pid, vals in zip(ids, class0_data) for val in vals
#                ] + [
#                    {'Participant': pid, 'Interval': val, 'Class': 'Chewing (1)'}
#                    for pid, vals in zip(ids, class1_data) for val in vals
#                ]
#
# df_box = pd.DataFrame(boxplot_data)
#
# if not df_box.empty:
#     sns.boxplot(x='Participant', y='Interval', hue='Class', data=df_box)
#     plt.title('Distribution of Zero Crossing Intervals by Participant and Class')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig('zero_crossing_intervals_boxplot.png', dpi=300)
#     plt.show()

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from definitions import ROOT_DIR

# Path setup (assuming these are defined in your project)
sourceFolder = ROOT_DIR
dataFolder_inlab_L = sourceFolder + '/datasets/inlab/left/'
dataFolder_inlab_R = sourceFolder + '/datasets/inlab/right/'
dataFolder_freeliving_L = sourceFolder + '/datasets/freeliving/'

participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def load_participant_data(participant_id, inlab_left_folder, inlab_right_folder, freeliving_folder):
    participant_data = {
        'inlab_left': None,
        'inlab_right': None,
        'freeliving_left': None
    }

    inlab_left = os.path.join(inlab_left_folder, f'participant_data_{participant_id}_left.csv')
    if os.path.exists(inlab_left):
        df_inlab_left = pd.read_csv(inlab_left)
        participant_data['inlab_left'] = {
            'prox_data': df_inlab_left['L_prox_sen_data'].values,
            'sequence_annotation': np.where(df_inlab_left['L_chew_cycle_annotation'].values > 0, 1, 0),
            'eating_annotation': [1 if x == 10000 else x for x in df_inlab_left['Eating_event_truth'].values]
        }

    inlab_right = os.path.join(inlab_right_folder, f'participant_data_{participant_id}_right.csv')
    if os.path.exists(inlab_right):
        df_inlab_right = pd.read_csv(inlab_right)
        participant_data['inlab_right'] = {
            'prox_data': df_inlab_right['R_prox_sen_data'].values,
            'sequence_annotation': np.where(df_inlab_right['R_chew_cycle_annotation'].values > 0, 1, 0),
            'eating_annotation': [1 if x == 10000 else x for x in df_inlab_right['Eating_event_truth'].values]
        }

    freeliving_left = os.path.join(freeliving_folder, f'participant_data_{participant_id}_left.csv')
    if os.path.exists(freeliving_left):
        df_freeliving_left = pd.read_csv(freeliving_left)
        participant_data['freeliving_left'] = {
            'prox_data': df_freeliving_left['L_prox_sen_data'].values,
            'sequence_annotation': df_freeliving_left['Annotation_Sequence_A'].values,
            'eating_annotation': [1 if x == 10000 else x for x in df_freeliving_left['Eating_event_truth'].values]
        }

    return participant_data


def combine_participant_sources(data):
    all_prox_data = []
    all_sequence_annotations = []
    all_eating_annotations = []

    for source, source_data in data.items():
        if source_data is not None:
            all_prox_data.extend(source_data['prox_data'])
            all_sequence_annotations.extend(source_data['sequence_annotation'])
            all_eating_annotations.extend(source_data['eating_annotation'])

    prox_data = np.array(all_prox_data)
    sequence_annotation = np.array(all_sequence_annotations)
    eating_annotation = np.array(all_eating_annotations)

    return prox_data, sequence_annotation, eating_annotation


def apply_ema_filter(data, alpha=0.55):
    """Apply Exponential Moving Average filter to the data"""
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = filtered[i - 1] - (alpha * (filtered[i - 1] - data[i]))
    return filtered


def standardize_frame(frame):
    """Standardize a frame by subtracting mean and dividing by standard deviation"""
    mean = np.mean(frame)
    std = np.std(frame)
    if std != 0:
        return (frame - mean) / std
    else:
        return frame - mean  # Just center if std is zero


def compute_zero_crossing_intervals(standardized_frame, amplitude_threshold=0.3):
    """Compute intervals between zero crossings that exceed the amplitude threshold"""
    crossings = []

    # Find zero crossings with significant amplitude change
    for i in range(1, len(standardized_frame)):
        if (standardized_frame[i - 1] * standardized_frame[i]) <= 0:  # Zero crossing
            amplitude_diff = abs(standardized_frame[i] - standardized_frame[i - 1])
            if amplitude_diff > amplitude_threshold:  # Apply threshold to filter noise
                crossings.append(i)

    # Calculate intervals between consecutive crossings
    if len(crossings) >= 2:
        intervals = np.diff(crossings)
        return intervals
    else:
        return np.array([])  # Return empty array if fewer than 2 crossings


def analyze_participant_frames(signal, labels, frame_size=128, overlap=64):
    """Analyze all frames from a participant's data and collect zero crossing intervals by class"""
    class0_intervals = []
    class1_intervals = []

    # Process frames with overlap
    for i in range(0, len(signal) - frame_size, overlap):
        frame = signal[i:i + frame_size]
        frame_labels = labels[i:i + frame_size]

        # Determine frame class (0 or 1) based on majority
        frame_class = 1 if np.mean(frame_labels) > 0.3 else 0

        # Apply EMA filter to frame
        filtered_frame = apply_ema_filter(frame)

        # Standardize the filtered frame
        standardized_frame = standardize_frame(filtered_frame)

        # Calculate zero crossing intervals
        intervals = compute_zero_crossing_intervals(standardized_frame)

        # Add intervals to corresponding class
        if intervals.size > 0:  # Only if we found intervals
            if frame_class == 0:
                class0_intervals.extend(intervals)
            else:
                class1_intervals.extend(intervals)

    return np.array(class0_intervals), np.array(class1_intervals)


# Function to calculate percentiles and plot histogram
def analyze_interval_distribution(intervals, class_label, participant_id):
    """Analyze the distribution of zero crossing intervals"""
    if intervals.size == 0:
        return None, None, None, None, None, None

    # Calculate statistics
    mean = np.mean(intervals)
    median = np.median(intervals)
    percentile_25 = np.percentile(intervals, 25)
    percentile_75 = np.percentile(intervals, 75)
    percentile_90 = np.percentile(intervals, 90)
    percentile_95 = np.percentile(intervals, 95)
    percentile_99 = np.percentile(intervals, 99)
    min_val = np.min(intervals)
    max_val = np.max(intervals)

    # Create histogram for this participant and class
    plt.figure(figsize=(10, 6))

    # Set appropriate bin size and range
    max_bin = min(100, max_val)  # Cap at 100 for better visualization

    # Create histogram
    hist, bins, _ = plt.hist(
        intervals,
        bins=np.arange(0, max_bin + 1, 1),  # Bins for each integer value up to max_bin
        alpha=0.7,
        color='blue' if class_label == 0 else 'red',
        label=f'Class {class_label} intervals'
    )

    # Add vertical lines for statistics
    plt.axvline(x=mean, color='green', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(x=median, color='black', linestyle='-', label=f'Median: {median:.2f}')
    plt.axvline(x=percentile_95, color='purple', linestyle=':', label=f'95th percentile: {percentile_95:.2f}')

    # Annotate the extreme values
    plt.text(max_bin * 0.7, max(hist) * 0.9, f'Max: {max_val:.2f}',
             bbox=dict(facecolor='yellow', alpha=0.5))

    # Add labels and title
    plt.xlabel('Zero Crossing Interval')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Zero Crossing Intervals - Participant {participant_id}, Class {class_label}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the histogram
    plt.tight_layout()
    plt.savefig(f'histogram_participant_{participant_id}_class_{class_label}.png', dpi=300)
    plt.close()

    return {
        'mean': mean,
        'median': median,
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'percentile_90': percentile_90,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'min': min_val,
        'max': max_val
    }


# Store results for all participants
results = []
all_class0_intervals = []
all_class1_intervals = []

# Process each participant
for participant_id in tqdm(participants, desc="Processing participants"):
    print(f"\nProcessing participant {participant_id}")

    # Load participant data
    participant_data = load_participant_data(
        participant_id,
        dataFolder_inlab_L,
        dataFolder_inlab_R,
        dataFolder_freeliving_L
    )

    # Combine data from all sources
    prox_data, sequence_annotation, _ = combine_participant_sources(participant_data)

    # Skip if no data available
    if len(prox_data) == 0:
        print(f"No data available for participant {participant_id}, skipping.")
        continue

    # Analyze frames to get intervals for each class
    class0_intervals, class1_intervals = analyze_participant_frames(prox_data, sequence_annotation)

    # Store all intervals for combined analysis
    all_class0_intervals.extend(class0_intervals)
    all_class1_intervals.extend(class1_intervals)

    # Analyze interval distributions
    class0_stats = analyze_interval_distribution(class0_intervals, 0, participant_id)
    class1_stats = analyze_interval_distribution(class1_intervals, 1, participant_id)

    # Print detailed statistics
    if class0_stats:
        print(f"Class 0 intervals: {len(class0_intervals)}")
        print(f"  Mean: {class0_stats['mean']:.2f}, Median: {class0_stats['median']:.2f}")
        print(
            f"  25th percentile: {class0_stats['percentile_25']:.2f}, 75th percentile: {class0_stats['percentile_75']:.2f}")
        print(
            f"  90th percentile: {class0_stats['percentile_90']:.2f}, 95th percentile: {class0_stats['percentile_95']:.2f}")
        print(f"  99th percentile: {class0_stats['percentile_99']:.2f}")
        print(f"  Min: {class0_stats['min']:.2f}, Max: {class0_stats['max']:.2f}")

    if class1_stats:
        print(f"Class 1 intervals: {len(class1_intervals)}")
        print(f"  Mean: {class1_stats['mean']:.2f}, Median: {class1_stats['median']:.2f}")
        print(
            f"  25th percentile: {class1_stats['percentile_25']:.2f}, 75th percentile: {class1_stats['percentile_75']:.2f}")
        print(
            f"  90th percentile: {class1_stats['percentile_90']:.2f}, 95th percentile: {class1_stats['percentile_95']:.2f}")
        print(f"  99th percentile: {class1_stats['percentile_99']:.2f}")
        print(f"  Min: {class1_stats['min']:.2f}, Max: {class1_stats['max']:.2f}")

    # Store results
    result_entry = {
        'participant_id': participant_id,
        'class0_interval_count': len(class0_intervals),
        'class1_interval_count': len(class1_intervals)
    }

    # Add class 0 statistics if available
    if class0_stats:
        result_entry.update({
            'class0_mean_interval': class0_stats['mean'],
            'class0_median_interval': class0_stats['median'],
            'class0_25th_percentile': class0_stats['percentile_25'],
            'class0_75th_percentile': class0_stats['percentile_75'],
            'class0_90th_percentile': class0_stats['percentile_90'],
            'class0_95th_percentile': class0_stats['percentile_95'],
            'class0_99th_percentile': class0_stats['percentile_99'],
            'class0_min_interval': class0_stats['min'],
            'class0_max_interval': class0_stats['max']
        })

    # Add class 1 statistics if available
    if class1_stats:
        result_entry.update({
            'class1_mean_interval': class1_stats['mean'],
            'class1_median_interval': class1_stats['median'],
            'class1_25th_percentile': class1_stats['percentile_25'],
            'class1_75th_percentile': class1_stats['percentile_75'],
            'class1_90th_percentile': class1_stats['percentile_90'],
            'class1_95th_percentile': class1_stats['percentile_95'],
            'class1_99th_percentile': class1_stats['percentile_99'],
            'class1_min_interval': class1_stats['min'],
            'class1_max_interval': class1_stats['max']
        })

    results.append(result_entry)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nResults summary:")
print(results_df)

# Save results to CSV
results_df.to_csv('zero_crossing_intervals_by_participant.csv', index=False)
print("Results saved to 'zero_crossing_intervals_by_participant.csv'")

# First create a global histogram of all intervals
plt.figure(figsize=(12, 8))
plt.hist(all_class0_intervals, bins=np.arange(0, 50, 1), alpha=0.5, color='blue', label='Non-Chewing (Class 0)')
plt.hist(all_class1_intervals, bins=np.arange(0, 50, 1), alpha=0.5, color='red', label='Chewing (Class 1)')
plt.xlabel('Zero Crossing Interval')
plt.ylabel('Frequency')
plt.title('Distribution of Zero Crossing Intervals Across All Participants')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('all_participants_interval_histogram.png', dpi=300)
plt.close()

# Now create a scatter plot with percentiles instead of min/max
plt.figure(figsize=(12, 8))

# Plot non-chewing (class 0) intervals with median
plt.scatter(
    results_df['participant_id'],
    results_df['class0_median_interval'],
    label='Non-Chewing (Class 0) Median',
    color='blue',
    marker='o',
    s=100,
    alpha=0.7
)

# Plot chewing (class 1) intervals with median
plt.scatter(
    results_df['participant_id'],
    results_df['class1_median_interval'],
    label='Chewing (Class 1) Median',
    color='red',
    marker='x',
    s=100
)

# Add error bars showing 25th to 75th percentile range (interquartile range)
for i, row in results_df.iterrows():
    # Class 0 error bars (25th to 75th percentile)
    if 'class0_25th_percentile' in row and not np.isnan(row['class0_25th_percentile']):
        plt.plot(
            [row['participant_id'], row['participant_id']],
            [row['class0_25th_percentile'], row['class0_75th_percentile']],
            color='blue',
            linewidth=2,
            alpha=0.7
        )

    # Class 1 error bars (25th to 75th percentile)
    if 'class1_25th_percentile' in row and not np.isnan(row['class1_25th_percentile']):
        plt.plot(
            [row['participant_id'], row['participant_id']],
            [row['class1_25th_percentile'], row['class1_75th_percentile']],
            color='red',
            linewidth=2,
            alpha=0.7
        )

    # Add thin lines for 5th to 95th percentile range
    if 'class0_95th_percentile' in row and not np.isnan(row['class0_95th_percentile']):
        plt.plot(
            [row['participant_id'], row['participant_id']],
            [row['class0_median_interval'] - (row['class0_95th_percentile'] - row['class0_median_interval']),
             row['class0_95th_percentile']],
            color='blue',
            linewidth=1,
            alpha=0.4
        )

    if 'class1_95th_percentile' in row and not np.isnan(row['class1_95th_percentile']):
        plt.plot(
            [row['participant_id'], row['participant_id']],
            [row['class1_median_interval'] - (row['class1_95th_percentile'] - row['class1_median_interval']),
             row['class1_95th_percentile']],
            color='red',
            linewidth=1,
            alpha=0.4
        )

# Add horizontal line showing typical chewing range (15-40)
plt.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Typical Chewing Range (15-40)')
plt.axhline(y=40, color='green', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel('Participant ID')
plt.ylabel('Zero Crossing Interval')
plt.title('Zero Crossing Intervals by Participant and Class (with Percentiles)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a text box with summary statistics
try:
    textstr = "\n".join([
        "Overall Statistics:",
        f"Class 0 Median: {results_df['class0_median_interval'].mean():.2f}",
        f"Class 1 Median: {results_df['class1_median_interval'].mean():.2f}",
        f"Class 0 IQR: {(results_df['class0_75th_percentile'] - results_df['class0_25th_percentile']).mean():.2f}",
        f"Class 1 IQR: {(results_df['class1_75th_percentile'] - results_df['class1_25th_percentile']).mean():.2f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
except KeyError:
    # Handle case where percentiles might not be available
    pass

plt.tight_layout()
plt.savefig('zero_crossing_intervals_by_participant_percentiles.png', dpi=300)
plt.show()

# Create a summary boxplot comparing all class 0 vs class 1 intervals
plt.figure(figsize=(10, 8))

# Convert to arrays for plotting
all_class0_intervals_array = np.array(all_class0_intervals)
all_class1_intervals_array = np.array(all_class1_intervals)

# Cap intervals for better visualization
max_display = 50
capped_class0 = np.clip(all_class0_intervals_array, 0, max_display)
capped_class1 = np.clip(all_class1_intervals_array, 0, max_display)

# Create boxplot
boxplot_data = {
    'Non-Chewing (0)': capped_class0,
    'Chewing (1)': capped_class1
}

plt.boxplot([capped_class0, capped_class1],
            labels=['Non-Chewing (0)', 'Chewing (1)'],
            showfliers=False,  # Hide outliers for cleaner visualization
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='darkblue', linewidth=2))

# Add labels and title
plt.ylabel('Zero Crossing Interval (capped at 50)')
plt.title('Comparison of Zero Crossing Intervals Between Classes (All Participants)')
plt.grid(True, alpha=0.3)

# Add statistics as text
class0_stats = {
    'mean': np.mean(all_class0_intervals_array),
    'median': np.median(all_class0_intervals_array),
    'p25': np.percentile(all_class0_intervals_array, 25),
    'p75': np.percentile(all_class0_intervals_array, 75),
    'p95': np.percentile(all_class0_intervals_array, 95)
}

class1_stats = {
    'mean': np.mean(all_class1_intervals_array),
    'median': np.median(all_class1_intervals_array),
    'p25': np.percentile(all_class1_intervals_array, 25),
    'p75': np.percentile(all_class1_intervals_array, 75),
    'p95': np.percentile(all_class1_intervals_array, 95)
}

stats_text = f"""
Class 0 Stats:
  Mean: {class0_stats['mean']:.2f}
  Median: {class0_stats['median']:.2f}
  IQR: {class0_stats['p75'] - class0_stats['p25']:.2f}
  95th percentile: {class0_stats['p95']:.2f}

Class 1 Stats:
  Mean: {class1_stats['mean']:.2f}
  Median: {class1_stats['median']:.2f}
  IQR: {class1_stats['p75'] - class1_stats['p25']:.2f}
  95th percentile: {class1_stats['p95']:.2f}
"""

plt.figtext(0.15, 0.02, stats_text, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

plt.tight_layout()
plt.savefig('zero_crossing_intervals_boxplot_all.png', dpi=300)
plt.show()