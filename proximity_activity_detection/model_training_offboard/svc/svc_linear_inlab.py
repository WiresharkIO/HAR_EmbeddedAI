import os
from definitions import ROOT_DIR

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


class LOPOFeatureSelectionSVC:

    def __init__(self, data_path, models_path, features_list=None):
        self.data_path = data_path
        self.models_path = models_path
        self.features_list = features_list
        os.makedirs(models_path, exist_ok=True)

        if features_list is None:
            # self.features_list = ['ZC_TOTAL', 'ZC_MEAN_INTERVAL', 'ZC_CHEW_INTERVALS', 'ZC_NONCHEW_INTERVALS',
            #                       'ZC_CHEW_RATIO', 'ZC_NONCHEW_RATIO',
            #                       'ZC_MEAN_AMP', 'ZC_MAX_AMP', 'ZC_MIN_AMP', 'ZC_STD_AMP',
            #                       'ZC_CHEW_MEAN_AMP', 'ZC_NONCHEW_MEAN_AMP', 'ZC_CHEW_STD_AMP', 'ZC_NONCHEW_STD_AMP',
            #
            #                       'TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN',
            #                       'TD_VARIANCE', 'TD_STD', 'TD_SKEW', 'TD_KURT', 'TD_IQR',
            #                       ]

            self.features_list = ['ZC_TOTAL', 'ZC_CHEW_INTERVALS', 'ZC_NONCHEW_INTERVALS',
                                  'ZC_CHEW_RATIO', 'TD_MAX', 'TD_MIN', 'TD_SKEW', 'TD_KURT']

            # 20 features - frame size 128
            # self.features_list = [
            #     'ZC_TOTAL', 'ZC_NONCHEW_INTERVALS', 'ZC_CHEW_RATIO', 'ZC_MAX_AMP',
            #     'ZC_STD_AMP', 'ZC_MEAN_AMP', 'ZC_CHEW_STD_AMP', 'ZC_CHEW_MEAN_AMP',
            #     'TD_MIN', 'TD_SKEW',
            #     'ZC_CHEW_INTERVALS', 'TD_KURT', 'ZC_MIN_AMP', 'TD_MAX_MIN', 'TD_STD',
            #     'TD_IQR', 'TD_RMS', 'TD_VARIANCE', 'TD_MAX', 'ZC_MEAN_INTERVAL'
            # ]

            #  feature list of inlab with frame size = 64
            # self.features_list = [
            #     'ZC_NONCHEW_INTERVALS', 'ZC_TOTAL', 'ZC_CHEW_RATIO', 'ZC_CHEW_STD_AMP',
            #     'ZC_CHEW_MEAN_AMP', 'ZC_STD_AMP', 'TD_VARIANCE', 'TD_STD',
                # 'TD_RMS', 'TD_SKEW', 'ZC_MEAN_INTERVAL', 'TD_MAX',
                # 'ZC_CHEW_INTERVALS', 'TD_MAX_MIN', 'TD_KURT', 'TD_IQR'
            # ]

    def load_and_prepare_data(self, participant_id, features_folder):

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        feature_files = [f for f in os.listdir(features_folder) if f.endswith('.csv')]

        for file in feature_files:
            file_path = os.path.join(features_folder, file)
            if f'sequence_features_{participant_id}.csv' == file:
                test_df = pd.read_csv(file_path)
                test_df = test_df.fillna(0)

            else:
                curr_df = pd.read_csv(file_path)
                curr_df = curr_df.fillna(0)
                train_df = pd.concat([train_df, curr_df], ignore_index=True)

        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")

        if not test_df.empty:
            print("\nClass distribution in test set:")
            print(test_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        if not train_df.empty:
            print("\nClass distribution in training set:")
            print(train_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        return train_df, test_df

    def evaluate_participant(self, y_true, y_pred, participant_id):

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')


        return accuracy, precision, recall, f1

    def train_and_evaluate(self, participants, output_path):
        results = []
        model_info = {}

        for participant_id in participants:
            print(f"\nProcessing participant {participant_id}")

            train_df, test_df = self.load_and_prepare_data(participant_id, self.data_path)

            X_train = train_df[self.features_list].values
            y_train = train_df['SEQUENCE_TRUTH_FRAME'].values

            X_test = test_df[self.features_list].values
            y_test = test_df['SEQUENCE_TRUTH_FRAME'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            svc = LinearSVC(C=0.1, dual="auto", loss='squared_hinge', penalty='l2', class_weight={0: 2, 1: 5},
                            max_iter=10000, verbose=True)
            svc.fit(X_train_scaled, y_train)

            y_pred = svc.predict(X_test_scaled)
            accuracy, precision, recall, f1 = self.evaluate_participant(y_test, y_pred, participant_id)

            results.append({
                'participant_id': participant_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            print(f"Results for participant {participant_id}:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"recall: {recall:.3f}")
            print(f"precision: {precision:.3f}")

            model_info[participant_id] = self.prepare_model_for_stm32(svc, scaler, participant_id, output_path)

            sklearn_model_path = os.path.join(model_info[participant_id]['participant_dir'],
                                              f'sklearn_model_{participant_id}.joblib')
            joblib.dump({'svc': svc, 'scaler': scaler, 'features': self.features_list}, sklearn_model_path)

        return pd.DataFrame(results), model_info

    def prepare_model_for_stm32(self, svc, scaler, participant_id, output_path):
        import tensorflow as tf
        import json

        participant_dir = os.path.join(output_path, f'participant_{participant_id}')
        os.makedirs(participant_dir, exist_ok=True)

        scaling_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }

        with open(os.path.join(participant_dir, f'scaler_params_{participant_id}.json'), 'w') as f:
            json.dump(scaling_params, f)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(self.features_list),), dtype=tf.float32, name='input'),
            tf.keras.layers.Dense(
                units=1,
                use_bias=True,
                kernel_initializer=tf.constant_initializer(svc.coef_[0]),
                bias_initializer=tf.constant_initializer(svc.intercept_[0]),
                name='svm_decision'
            ),
			
            tf.keras.layers.Activation('sigmoid', name='output')
        ])


        model.save(os.path.join(participant_dir, f'keras_model_{participant_id}.h5'))

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
		
        converter.optimizations = []
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()

        tflite_path = os.path.join(participant_dir, f"svm_model_{participant_id}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {tflite_path}")

        try:
            import onnx
            import tf2onnx

            input_signature = [tf.TensorSpec(shape=(None, len(self.features_list)), dtype=tf.float32)]

            onnx_model, _ = tf2onnx.convert.from_function(
                model,
                input_signature=input_signature,
                opset=12
            )

            onnx_path = os.path.join(participant_dir, f"svm_model_{participant_id}.onnx")
            onnx.save(onnx_model, onnx_path)
            print(f"ONNX model saved to {onnx_path}")
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
            print("Continuing without ONNX model...")
            onnx_path = None

        with open(os.path.join(participant_dir, f'svm_model_{participant_id}.h'), 'w') as f:
            f.write(f"#ifndef SVM_MODEL_{participant_id}_H\n")
            f.write(f"#define SVM_MODEL_{participant_id}_H\n\n")
            f.write(f"#define NUM_FEATURES {len(self.features_list)}\n\n")

            f.write("// SVM coefficients\n")
            f.write("static const float svm_coefficients[NUM_FEATURES] = {\n")
            for i, coef in enumerate(svc.coef_[0]):
                f.write(f"    {coef:.8f}f" + ("," if i < len(svc.coef_[0]) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write(f"static const float svm_intercept = {svc.intercept_[0]:.8f}f;\n\n")

            f.write("#endif // SVM_MODEL_{participant_id}_H\n")

        with open(os.path.join(participant_dir, f'svm_features_{participant_id}.h'), 'w') as f:
            f.write(f"#ifndef SVM_FEATURES_{participant_id}_H\n")
            f.write(f"#define SVM_FEATURES_{participant_id}_H\n\n")

            f.write(f"#define NUM_FEATURES {len(self.features_list)}\n\n")

            f.write("// Scaling parameters\n")
            f.write("static const float feature_means[NUM_FEATURES] = {\n")
            for i, mean in enumerate(scaler.mean_):
                f.write(f"    {mean:.8f}f" + ("," if i < len(scaler.mean_) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write("static const float feature_scales[NUM_FEATURES] = {\n")
            for i, scale in enumerate(scaler.scale_):
                f.write(f"    {scale:.8f}f" + ("," if i < len(scaler.scale_) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write("#endif // SVM_FEATURES_{participant_id}_H\n")

        with open(os.path.join(participant_dir, f'svm_complete_{participant_id}.h'), 'w') as f:
            f.write(f"#ifndef SVM_COMPLETE_{participant_id}_H\n")
            f.write(f"#define SVM_COMPLETE_{participant_id}_H\n\n")

            f.write(f"#define NUM_FEATURES {len(self.features_list)}\n\n")

            f.write("// Feature names for reference\n/*\n")
            for i, feature in enumerate(self.features_list):
                f.write(f"    {i}: {feature}\n")
            f.write("*/\n\n")

            f.write("// Scaling parameters\n")
            f.write("static const float feature_means[NUM_FEATURES] = {\n")
            for i, mean in enumerate(scaler.mean_):
                f.write(f"    {mean:.8f}f" + ("," if i < len(scaler.mean_) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write("static const float feature_scales[NUM_FEATURES] = {\n")
            for i, scale in enumerate(scaler.scale_):
                f.write(f"    {scale:.8f}f" + ("," if i < len(scaler.scale_) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write("// SVM model parameters\n")
            f.write("static const float svm_coefficients[NUM_FEATURES] = {\n")
            for i, coef in enumerate(svc.coef_[0]):
                f.write(f"    {coef:.8f}f" + ("," if i < len(svc.coef_[0]) - 1 else "") + "\n")
            f.write("};\n\n")

            f.write(f"static const float svm_intercept = {svc.intercept_[0]:.8f}f;\n\n")

            f.write("// Inference function\n")
            f.write("static inline float svm_predict(const float* features) {\n")
            f.write("    float scaled_features[NUM_FEATURES];\n")
            f.write("    float result = 0.0f;\n\n")

            f.write("    // Apply scaling\n")
            f.write("    for (int i = 0; i < NUM_FEATURES; i++) {\n")
            f.write("        scaled_features[i] = (features[i] - feature_means[i]) / feature_scales[i];\n")
            f.write("    }\n\n")

            f.write("    // Apply SVM decision function\n")
            f.write("    for (int i = 0; i < NUM_FEATURES; i++) {\n")
            f.write("        result += scaled_features[i] * svm_coefficients[i];\n")
            f.write("    }\n")
            f.write("    result += svm_intercept;\n\n")

            f.write("    return result;\n")
            f.write("}\n\n")

            f.write("// Binary classification function\n")
            f.write("static inline int svm_classify(const float* features) {\n")
            f.write("    return svm_predict(features) > 0.0f ? 1 : 0;\n")
            f.write("}\n\n")

            f.write("#endif // SVM_COMPLETE_{participant_id}_H\n")

        return {
            'tflite_path': tflite_path,
            'onnx_path': onnx_path,
            'participant_dir': participant_dir
        }

if __name__ == "__main__":
    sourceFolder = ROOT_DIR
    # featuresFolder = sourceFolder + '//features//sequence//preprocessing//feature_extracted//EMA//inlab_feature_extraction//'
    # featuresFolder = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/64_feature_extraction/inlab_feature_extraction/'
    featuresFolder = sourceFolder + '/features/sequence/preprocessing/feature_extracted/EMA/128_feature_extraction/inlab_feature_extraction/'
    savedModels = sourceFolder + '//models//sequence//SVM//inference//svc_linear_inlab//128_model_files//'

    participants = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    lopo_svc = LOPOFeatureSelectionSVC(featuresFolder, savedModels)
    results, model_info = lopo_svc.train_and_evaluate(participants, savedModels)

    print("\nOverall Results:")
    print(f"Average Accuracy: {results['accuracy'].mean():.3f} ± {results['accuracy'].std():.3f}")
    print(f"Average F1 Score: {results['f1_score'].mean():.3f} ± {results['f1_score'].std():.3f}")
    print(f"Average Recall: {results['recall'].mean():.3f} ± {results['recall'].std():.3f}")
    print(f"Average Precision: {results['precision'].mean():.3f} ± {results['precision'].std():.3f}")

    print("\nModel Export Information:")
    for participant_id, info in model_info.items():
        print(f"Participant {participant_id}: Models saved to {info['participant_dir']}")