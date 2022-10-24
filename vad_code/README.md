

## TODO
- [ ] Run grid-search
- [ ] Try 20s segments for testing, compare with 400s
- [ ] Train with weighted cross-entropy loss
- [ ] Process AVA data, and eval


- [x] Visualize multilingual test data predictions & labels to see where the model falls short
- [x] Prepare and cut AMI manifest for train/dev/test
- [x] Prepare and cut isci for training
- [x] Prepare and cut fisher 2004 manifest for train
- [x] Prepare and cut fisher 2005 manifest for train
- [x] Prepare and cut ch120-Ch109 manifest for test
- [x] Prepare and cut ch120-moved for dev


## Changed Files
- nemo.collections.asr.data.audio_to_label
- nemo.collections.asr.data.audio_to_label_dataset




```shell
python tools/clean_manifest.py manifests_draco/french_train_40ms.json \

```


## Data Processing
1. Generate manifest with: audio_filepath, duration, offset
2. For each manifest item, load RTTM and obtain frame-level labels


- Noise aug fail with worker > 0, same error message as before
  - "Terminating: fork() called from a process already using GNU OpenMP, this is unsafe."
- No noise aug works fine with workers = 1, but fail after 3 epochs with worker=4,
  - "zipfile.BadZipFile: Bad magic number for file header" ---- happened in AudioSegment.from_file()->librosa.core.resample()




########################### proj_name="Stream_VAD", exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:116] ================= french_test_20ms =================
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:117] AUROC: 0.9558
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.83      0.50      0.63    568456
               1       0.91      0.98      0.94   2904042

        accuracy                           0.90   3472498
       macro avg       0.87      0.74      0.79   3472498
    weighted avg       0.90      0.90      0.89   3472498

[NeMo I 2022-10-04 14:56:10 infer_vad_multi:125] Detection Error Rate: DetER=11.8504, False Alarm=10.2319, Miss=1.6185
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:116] ================= german_test_20ms =================
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:117] AUROC: 0.9634
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.85      0.73      0.78   2893332
               1       0.89      0.94      0.92   6700110

        accuracy                           0.88   9593442
       macro avg       0.87      0.84      0.85   9593442
    weighted avg       0.88      0.88      0.88   9593442

[NeMo I 2022-10-04 14:56:32 infer_vad_multi:125] Detection Error Rate: DetER=16.0786, False Alarm=9.7779, Miss=6.3007
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:116] ================= mandarin_test_20ms =================
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:117] AUROC: 0.8878
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.84      0.43      0.57    365268
               1       0.84      0.97      0.90   1081786

        accuracy                           0.84   1447054
       macro avg       0.84      0.70      0.74   1447054
    weighted avg       0.84      0.84      0.82   1447054

[NeMo I 2022-10-04 14:56:35 infer_vad_multi:125] Detection Error Rate: DetER=21.3148, False Alarm=18.6094, Miss=2.7054
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:116] ================= russian_test_20ms =================
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:117] AUROC: 0.9605
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.86      0.63      0.73   1389584
               1       0.89      0.97      0.93   4339820

        accuracy                           0.88   5729404
       macro avg       0.87      0.80      0.83   5729404
    weighted avg       0.88      0.88      0.88   5729404

[NeMo I 2022-10-04 14:56:46 infer_vad_multi:125] Detection Error Rate: DetER=14.1586, False Alarm=10.5631, Miss=3.5955
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:116] ================= spanish_test_20ms =================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:117] AUROC: 0.9533
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.83      0.49      0.62    526170
               1       0.92      0.98      0.95   3078748

        accuracy                           0.91   3604918
       macro avg       0.87      0.74      0.78   3604918
    weighted avg       0.91      0.91      0.90   3604918

[NeMo I 2022-10-04 14:56:53 infer_vad_multi:125] Detection Error Rate: DetER=10.4260, False Alarm=9.0447, Miss=1.3813
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:133]  DetER=10.4260, False Alarm=9.0447, Miss=1.3813
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:139] AUROC: 0.9573
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:141]
                  precision    recall  f1-score   support

               0       0.85      0.64      0.73   5742810
               1       0.89      0.96      0.93  18104506

        accuracy                           0.89  23847316
       macro avg       0.87      0.80      0.83  23847316
    weighted avg       0.88      0.89      0.88  23847316

[NeMo I 2022-10-10 13:34:03 infer_vad_multi:100] ====================================================
[NeMo I 2022-10-10 13:34:03 infer_vad_multi:101] Finalizing individual results...
[NeMo I 2022-10-10 13:34:13 infer_vad_multi:116] ================= ami_eval_10ms =================
[NeMo I 2022-10-10 13:34:13 infer_vad_multi:117] AUROC: 0.9466
[NeMo I 2022-10-10 13:34:13 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-10 13:34:13 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.55      0.88      0.68    762396
               1       0.98      0.87      0.92   4238736

        accuracy                           0.87   5001132
       macro avg       0.76      0.87      0.80   5001132
    weighted avg       0.91      0.87      0.88   5001132

[NeMo I 2022-10-10 13:34:13 infer_vad_multi:125] Detection Error Rate: DetER=15.1767, False Alarm=1.7453, Miss=13.4314
[NeMo I 2022-10-10 13:34:13 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:116] ================= ch120_CH109_10ms =================
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:117] AUROC: 0.8774
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.72      0.38      0.50    799798
               1       0.91      0.98      0.94   5227642

        accuracy                           0.90   6027440
       macro avg       0.81      0.68      0.72   6027440
    weighted avg       0.89      0.90      0.88   6027440

[NeMo I 2022-10-10 13:34:25 infer_vad_multi:125] Detection Error Rate: DetER=11.6203, False Alarm=9.6153, Miss=2.0050
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:133]  DetER=11.6203, False Alarm=9.6153, Miss=2.0050
[NeMo I 2022-10-10 13:34:25 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-10 13:34:49 infer_vad_multi:139] AUROC: 0.8904
[NeMo I 2022-10-10 13:34:49 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-10 13:34:49 infer_vad_multi:141]
                  precision    recall  f1-score   support

               0       0.59      0.62      0.61   1562194
               1       0.94      0.93      0.93   9466378

        accuracy                           0.89  11028572
       macro avg       0.76      0.78      0.77  11028572
    weighted avg       0.89      0.89      0.89  11028572










################################# marblenet_3x2x64_mandarin_40ms_all
[NeMo I 2022-10-10 12:15:25 infer_vad_multi:100] ====================================================
[NeMo I 2022-10-10 12:15:25 infer_vad_multi:101] Finalizing individual results...
[NeMo I 2022-10-10 12:15:35 infer_vad_multi:116] ================= ami_eval_10ms =================
[NeMo I 2022-10-10 12:15:35 infer_vad_multi:117] AUROC: 0.9500
[NeMo I 2022-10-10 12:15:35 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-10 12:15:35 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.56      0.92      0.70    762396
               1       0.98      0.87      0.92   4238736

        accuracy                           0.88   5001132
       macro avg       0.77      0.90      0.81   5001132
    weighted avg       0.92      0.88      0.89   5001132

[NeMo I 2022-10-10 12:15:35 infer_vad_multi:125] Detection Error Rate: DetER=13.5513, False Alarm=1.3435, Miss=12.2078
[NeMo I 2022-10-10 12:15:35 infer_vad_multi:126] ==========================================

[NeMo I 2022-10-10 12:15:47 infer_vad_multi:116] ================= ch120_CH109_10ms =================
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:117] AUROC: 0.9320
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.84      0.37      0.52    799798
               1       0.91      0.99      0.95   5227642

        accuracy                           0.91   6027440
       macro avg       0.88      0.68      0.73   6027440
    weighted avg       0.90      0.91      0.89   6027440

[NeMo I 2022-10-10 12:15:47 infer_vad_multi:125] Detection Error Rate: DetER=11.1276, False Alarm=10.3724, Miss=0.7552
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:126] ==========================================


[NeMo I 2022-10-10 12:15:47 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:133]  DetER=11.1276, False Alarm=10.3724, Miss=0.7552
[NeMo I 2022-10-10 12:15:47 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-10 12:16:10 infer_vad_multi:139] AUROC: 0.9289
[NeMo I 2022-10-10 12:16:10 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-10 12:16:10 infer_vad_multi:141]
                  precision    recall  f1-score   support

               0       0.63      0.64      0.63   1562194
               1       0.94      0.94      0.94   9466378

        accuracy                           0.89  11028572
       macro avg       0.78      0.79      0.79  11028572
    weighted avg       0.90      0.89      0.90  11028572


[NeMo I 2022-10-12 10:39:53 infer_vad_multi:116] ================= ava_eval_10ms =================
[NeMo I 2022-10-12 10:39:53 infer_vad_multi:117] AUROC: 0.8691
[NeMo I 2022-10-12 10:39:53 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-12 10:39:53 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.80      0.69      0.74   5061417
               1       0.76      0.85      0.80   5919315

        accuracy                           0.78  10980732
       macro avg       0.78      0.77      0.77  10980732
    weighted avg       0.78      0.78      0.78  10980732
[NeMo I 2022-10-12 10:39:53 infer_vad_multi:125] Detection Error Rate: DetER=40.0174, False Alarm=24.4071, Miss=15.6102





########################## split=20s , window=0

[NeMo I 2022-10-12 11:11:38 infer_vad_multi:116] ================= ava_eval_10ms =================
[NeMo I 2022-10-12 11:11:38 infer_vad_multi:117] AUROC: 0.7898
[NeMo I 2022-10-12 11:11:38 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-12 11:11:38 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.71      0.63      0.67   5065807
               1       0.71      0.78      0.75   5925173

        accuracy                           0.71  10990980
       macro avg       0.71      0.71      0.71  10990980
    weighted avg       0.71      0.71      0.71  10990980

[NeMo I 2022-10-12 11:11:38 infer_vad_multi:125] Detection Error Rate: DetER=51.2626, False Alarm=29.3654, Miss=21.8971
[NeMo I 2022-10-12 11:11:38 infer_vad_multi:126] ==========================================


[NeMo I 2022-10-12 11:12:31 infer_vad_multi:116] ================= ami_eval_10ms =================
[NeMo I 2022-10-12 11:12:31 infer_vad_multi:117] AUROC: 0.8130
[NeMo I 2022-10-12 11:12:31 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-12 11:12:31 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.41      0.69      0.52    763449
               1       0.94      0.83      0.88   4242435

        accuracy                           0.80   5005884
       macro avg       0.67      0.76      0.70   5005884
    weighted avg       0.86      0.80      0.82   5005884

[NeMo I 2022-10-12 11:12:31 infer_vad_multi:125] Detection Error Rate: DetER=22.2158, False Alarm=5.3569, Miss=16.8589
[NeMo I 2022-10-12 11:12:31 infer_vad_multi:126] ==========================================


[NeMo I 2022-10-12 11:13:38 infer_vad_multi:116] ================= ch120_CH109_10ms =================
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:117] AUROC: 0.7663
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.67      0.31      0.43    799837
               1       0.90      0.98      0.94   5233369

        accuracy                           0.89   6033206
       macro avg       0.79      0.64      0.68   6033206
    weighted avg       0.87      0.89      0.87   6033206

[NeMo I 2022-10-12 11:13:38 infer_vad_multi:125] Detection Error Rate: DetER=12.6284, False Alarm=10.9186, Miss=1.7098
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:126] ==========================================


[NeMo I 2022-10-12 11:13:38 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:133]  DetER=12.6284, False Alarm=10.9186, Miss=1.7098
[NeMo I 2022-10-12 11:13:38 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-12 11:18:49 infer_vad_multi:139] AUROC: 0.8034
[NeMo I 2022-10-12 11:18:49 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-12 11:18:49 infer_vad_multi:141]
                  precision    recall  f1-score   support

               0       0.65      0.60      0.62   6629093
               1       0.83      0.86      0.85  15400977

        accuracy                           0.78  22030070
       macro avg       0.74      0.73      0.73  22030070
    weighted avg       0.78      0.78      0.78  22030070




{'min_duration_off': 0.0, 'min_duration_on': 0.15, 'offset': 0.9, 'onset': 0.6, 'pad_offset': 0.3, 'pad_onset': 0.4, 'filter_speech_first': True}, {'DetER (%)': 7.097828212929912, 'FA (%)': 4.51764277262578, 'MISS (%)': 2.5801854403041333}
[NeMo I 2022-10-18 10:31:47 infer_vad_multi:116] ================= ami_eval_10ms =================
[NeMo I 2022-10-18 10:31:47 infer_vad_multi:117] AUROC: 0.9500
[NeMo I 2022-10-18 10:31:47 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-18 10:31:47 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.56      0.92      0.70    762396
               1       0.98      0.87      0.92   4238736

        accuracy                           0.88   5001132
       macro avg       0.77      0.90      0.81   5001132
    weighted avg       0.92      0.88      0.89   5001132

[NeMo I 2022-10-18 10:31:47 infer_vad_multi:125] Detection Error Rate: DetER=7.9298, False Alarm=5.0006, Miss=2.9293





with noise aug, 22 epochs
[NeMo I 2022-10-19 16:03:05 infer_vad_multi:100] ====================================================
[NeMo I 2022-10-19 16:03:05 infer_vad_multi:101] Finalizing individual results...
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:116] ================= ava_eval_10ms =================
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:117] AUROC: 0.8749
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:119]               precision    recall  f1-score   support
    
               0       0.85      0.42      0.56   5061417
               1       0.65      0.93      0.77   5919315
    
        accuracy                           0.70  10980732
       macro avg       0.75      0.68      0.67  10980732
    weighted avg       0.74      0.70      0.67  10980732
    
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:125] Detection Error Rate: DetER=66.6590, False Alarm=66.1520, Miss=0.5070
[NeMo I 2022-10-19 16:22:07 infer_vad_multi:126] ==========================================
    
    
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:116] ================= ami_eval_10ms =================
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:117] AUROC: 0.9398
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:119]               precision    recall  f1-score   support
    
               0       0.58      0.89      0.70    762396
               1       0.98      0.88      0.93   4238736
    
        accuracy                           0.88   5001132
       macro avg       0.78      0.89      0.81   5001132
    weighted avg       0.92      0.88      0.89   5001132
    
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:125] Detection Error Rate: DetER=8.1966, False Alarm=5.4654, Miss=2.7311
[NeMo I 2022-10-19 16:31:54 infer_vad_multi:126] ==========================================
    
    
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:116] ================= ch120_CH109_10ms =================
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:117] AUROC: 0.9090
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:119]               precision    recall  f1-score   support
    
               0       0.85      0.32      0.47    799798
               1       0.91      0.99      0.95   5227642
    
        accuracy                           0.90   6027440
       macro avg       0.88      0.66      0.71   6027440
    weighted avg       0.90      0.90      0.88   6027440
    
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:125] Detection Error Rate: DetER=14.0187, False Alarm=13.7322, Miss=0.2865
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:126] ==========================================
    
    
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:133]  DetER=14.0187, False Alarm=13.7322, Miss=0.2865
[NeMo I 2022-10-19 16:37:59 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-19 17:12:52 infer_vad_multi:139] AUROC: 0.8863
[NeMo I 2022-10-19 17:12:52 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-19 17:12:52 infer_vad_multi:141] 
                  precision    recall  f1-score   support
    
               0       0.77      0.46      0.58   6623611
               1       0.80      0.94      0.87  15385693
    
        accuracy                           0.80  22009304
       macro avg       0.79      0.70      0.72  22009304
    weighted avg       0.79      0.80      0.78  22009304