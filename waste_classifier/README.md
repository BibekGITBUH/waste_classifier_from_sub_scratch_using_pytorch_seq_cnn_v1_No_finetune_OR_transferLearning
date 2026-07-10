# Smart Waste Image Classifier — Phase 1: Custom CNN

## Project structure
```
waste_classifier/
├── config.py           # paths, hyperparameters, class names
├── prepare_data.py     # splits datasets/final/<class> into train/val/test
├── dataset.py           # transforms, augmentation, dataloaders
├── models/
│   └── custom_cnn.py    # baseline CNN built from scratch
├── train.py             # training loop, early stopping, checkpointing
├── evaluate.py           # accuracy/precision/recall/F1/confusion matrix
├── checkpoints/          # saved model weights (.pt)
├── outputs/              # metrics, plots, training history
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run order
1. Make sure your organized dataset from the download script lives at
   `datasets/final/<class_name>/*.jpg` with classes:
   `glass, metal, organic, paper, plastic`

2. Split into train/val/test (80/10/10):
   ```bash
   python prepare_data.py
   ```

3. (Optional) sanity-check the dataloaders:
   ```bash
   python dataset.py
   ```

4. Train the custom CNN:
   ```bash
   python train.py --model custom_cnn
   ```
   This saves the best checkpoint to `checkpoints/custom_cnn_best.pt` and
   training curves to `outputs/custom_cnn_history.json`.

5. Evaluate on the test set:
   ```bash
   python evaluate.py --model custom_cnn
   ```
   This prints accuracy/precision/recall/F1 per class, and saves:
   - `outputs/custom_cnn_classification_report.json`
   - `outputs/custom_cnn_confusion_matrix.png`

## What's next (Phase 2)
Once the custom CNN baseline is trained and evaluated, we'll add:
- `models/pretrained.py` with MobileNetV2 and ResNet50 (transfer learning,
  fine-tuned on this dataset)
- A comparison script/table (custom CNN vs MobileNetV2 vs ResNet50)
- A Streamlit demo app for interactive predictions

## Notes on class balance
If `prepare_data.py` prints very different image counts across classes,
consider using a weighted loss (`nn.CrossEntropyLoss(weight=...)`) or
weighted sampling — flag this to me once you've run it and I'll add that in.




pip install -r requirements.txt
python prepare_data.py
python train.py --model custom_cnn
python evaluate.py --model custom_cnn



















Epoch  1/25 | train_loss=1.3658 train_acc=0.4224 | val_loss=1.2757 val_acc=0.4697 | 549.1s
Epoch  2/25 | train_loss=1.2681 train_acc=0.4773 | val_loss=1.1993 val_acc=0.5073 | 535.2s
Epoch  3/25 | train_loss=1.2099 train_acc=0.5100 | val_loss=1.2694 val_acc=0.5012 | 554.8s
Epoch  4/25 | train_loss=1.2140 train_acc=0.4981 | val_loss=1.1847 val_acc=0.4976 | 530.9s
Epoch  5/25 | train_loss=1.1722 train_acc=0.5170 | val_loss=1.2864 val_acc=0.5061 | 526.5s
Epoch  6/25 | train_loss=1.0899 train_acc=0.5607 | val_loss=1.0445 val_acc=0.5944 | 526.2s
Epoch  7/25 | train_loss=1.0634 train_acc=0.5686 | val_loss=1.0148 val_acc=0.6029 | 527.5s
Epoch  8/25 | train_loss=1.0577 train_acc=0.5794 | val_loss=1.1664 val_acc=0.5472 | 525.3s
Epoch  9/25 | train_loss=1.0281 train_acc=0.5980 | val_loss=0.9956 val_acc=0.6077 | 521.3s
Epoch 10/25 | train_loss=0.9803 train_acc=0.6164 | val_loss=1.0631 val_acc=0.5969 | 516.0s
Epoch 11/25 | train_loss=0.9623 train_acc=0.6309 | val_loss=0.9052 val_acc=0.6622 | 515.6s
Epoch 12/25 | train_loss=0.9301 train_acc=0.6394 | val_loss=0.8596 val_acc=0.6695 | 516.1s
Epoch 13/25 | train_loss=0.9273 train_acc=0.6487 | val_loss=0.8165 val_acc=0.6828 | 517.7s
Epoch 14/25 | train_loss=0.8819 train_acc=0.6677 | val_loss=0.9861 val_acc=0.6162 | 514.5s
Epoch 15/25 | train_loss=0.8820 train_acc=0.6698 | val_loss=0.9102 val_acc=0.6501 | 513.3s
Epoch 16/25 | train_loss=0.8487 train_acc=0.6789 | val_loss=0.8583 val_acc=0.6683 | 514.5s
Epoch 17/25 | train_loss=0.7857 train_acc=0.7107 | val_loss=0.7526 val_acc=0.7094 | 516.3s
Epoch 18/25 | train_loss=0.7583 train_acc=0.7186 | val_loss=0.7002 val_acc=0.7312 | 570.3s
Epoch 19/25 | train_loss=0.7406 train_acc=0.7237 | val_loss=0.7996 val_acc=0.7167 | 577.7s
Epoch 20/25 | train_loss=0.7294 train_acc=0.7346 | val_loss=0.6927 val_acc=0.7458 | 577.8s
Epoch 21/25 | train_loss=0.7159 train_acc=0.7355 | val_loss=0.7019 val_acc=0.7373 | 578.2s
Epoch 22/25 | train_loss=0.7068 train_acc=0.7383 | val_loss=0.6017 val_acc=0.7893 | 577.8s
Epoch 23/25 | train_loss=0.6980 train_acc=0.7429 | val_loss=0.6383 val_acc=0.7651 | 582.2s
Epoch 24/25 | train_loss=0.6831 train_acc=0.7466 | val_loss=0.6696 val_acc=0.7482 | 582.3s
Epoch 25/25 | train_loss=0.6651 train_acc=0.7583 | val_loss=0.6416 val_acc=0.7446 | 578.5s

Best val accuracy: 0.7893
Saved best checkpoint to: checkpoints\custom_cnn_best.pt
Saved training history to: outputs\custom_cnn_history.json

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python evaluate.py --model custom_cnn
C:\Users\BIBEK\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py:1095: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)

=== custom_cnn — Test Set Results ===
Accuracy: 0.7909

Class       Precision   Recall      F1          Support   
glass       0.8441      0.6978      0.7640      225       
metal       0.7039      0.7985      0.7483      134       
organic     0.8250      0.9296      0.8742      71        
paper       0.8214      0.8342      0.8278      193       
plastic     0.7661      0.7990      0.7822      209       

Macro avg   0.7921      0.8118      0.7993      

Saved full classification report to: outputs\custom_cnn_classification_report.json
Saved confusion matrix plot to: outputs\custom_cnn_confusion_matrix.png

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python predict.py --image D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier\datasets\split\test\metal\trashnet_172.jpg
✓ Loaded model from: checkpoints\custom_cnn_best.pt
✓ Loaded image: D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier\datasets\split\test\metal\trashnet_172.jpg (size: (512, 384))

==================================================
Predicted Class: metal
Confidence: 92.23%
==================================================

Class Probabilities:
  glass      2.14% 
  metal      92.23% ███████████████████████████
  organic    0.16% 
  paper      5.31% █
  plastic    0.15% 

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python predict.py --folder datasets/split/self_test
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 29 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
WhatsApp Image 2026-07-05 at 11.03.42 AM.jpeg                plastic      48.93%
WhatsApp Image 2026-07-05 at 11.03.43 AM (1).jpeg            plastic      55.64%
WhatsApp Image 2026-07-05 at 11.03.43 AM (2).jpeg            paper        97.70%
WhatsApp Image 2026-07-05 at 11.03.43 AM (3).jpeg            paper        80.90%
WhatsApp Image 2026-07-05 at 11.03.43 AM (4).jpeg            plastic      36.93%
WhatsApp Image 2026-07-05 at 11.03.43 AM (5).jpeg            plastic      47.57%
WhatsApp Image 2026-07-05 at 11.03.43 AM (6).jpeg            plastic      79.84%
WhatsApp Image 2026-07-05 at 11.03.43 AM (7).jpeg            plastic      72.18%
WhatsApp Image 2026-07-05 at 11.03.43 AM (8).jpeg            glass        58.52%
WhatsApp Image 2026-07-05 at 11.03.43 AM (9).jpeg            plastic      56.62%
WhatsApp Image 2026-07-05 at 11.03.43 AM.jpeg                plastic      80.71%
WhatsApp Image 2026-07-05 at 11.03.44 AM (1).jpeg            glass        49.58%
WhatsApp Image 2026-07-05 at 11.03.44 AM (10).jpeg           plastic      75.92%
WhatsApp Image 2026-07-05 at 11.03.44 AM (11).jpeg           metal        38.29%
WhatsApp Image 2026-07-05 at 11.03.44 AM (12).jpeg           plastic      78.52%
WhatsApp Image 2026-07-05 at 11.03.44 AM (13).jpeg           plastic      64.63%
WhatsApp Image 2026-07-05 at 11.03.44 AM (14).jpeg           paper        54.09%
WhatsApp Image 2026-07-05 at 11.03.44 AM (15).jpeg           paper        93.33%
WhatsApp Image 2026-07-05 at 11.03.44 AM (16).jpeg           plastic      65.68%
WhatsApp Image 2026-07-05 at 11.03.44 AM (17).jpeg           glass        63.31%
WhatsApp Image 2026-07-05 at 11.03.44 AM (2).jpeg            glass        50.38%
WhatsApp Image 2026-07-05 at 11.03.44 AM (3).jpeg            plastic      84.94%
WhatsApp Image 2026-07-05 at 11.03.44 AM (4).jpeg            plastic      76.85%
WhatsApp Image 2026-07-05 at 11.03.44 AM (5).jpeg            glass        38.99%
WhatsApp Image 2026-07-05 at 11.03.44 AM (6).jpeg            paper        44.58%
WhatsApp Image 2026-07-05 at 11.03.44 AM (7).jpeg            metal        48.58%
WhatsApp Image 2026-07-05 at 11.03.44 AM (8).jpeg            plastic      57.72%
WhatsApp Image 2026-07-05 at 11.03.44 AM (9).jpeg            metal        34.78%
WhatsApp Image 2026-07-05 at 11.03.44 AM.jpeg                paper        92.39%
====================================================================================

✓ Processing complete! Total: 29 images
✓ Results saved to: datasets\split\self_test\predictions_20260705_111018.csv

Summary by Class:
  glass           5 images
  metal           3 images
  organic         0 images
  paper           6 images
  plastic        15 images

Average Confidence: 63.04%

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python predict.py --folder datasets/split/self_test
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 14 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
WhatsApp Image 2026-07-05 at 11.23.28 AM (1).jpeg            plastic      50.09%
WhatsApp Image 2026-07-05 at 11.23.28 AM.jpeg                paper        93.16%
WhatsApp Image 2026-07-05 at 11.23.29 AM (1).jpeg            glass        50.42%
WhatsApp Image 2026-07-05 at 11.23.29 AM (10).jpeg           glass        84.31%
WhatsApp Image 2026-07-05 at 11.23.29 AM (11).jpeg           glass        46.93%
WhatsApp Image 2026-07-05 at 11.23.29 AM (2).jpeg            metal        66.84%
WhatsApp Image 2026-07-05 at 11.23.29 AM (3).jpeg            organic      97.72%
WhatsApp Image 2026-07-05 at 11.23.29 AM (4).jpeg            metal        53.05%
WhatsApp Image 2026-07-05 at 11.23.29 AM (5).jpeg            plastic      78.12%
WhatsApp Image 2026-07-05 at 11.23.29 AM (6).jpeg            plastic      57.61%
WhatsApp Image 2026-07-05 at 11.23.29 AM (7).jpeg            paper        69.86%
WhatsApp Image 2026-07-05 at 11.23.29 AM (8).jpeg            glass        74.41%
WhatsApp Image 2026-07-05 at 11.23.29 AM (9).jpeg            glass        87.05%
WhatsApp Image 2026-07-05 at 11.23.29 AM.jpeg                plastic      50.83%
====================================================================================

✓ Processing complete! Total: 14 images
✓ Results saved to: datasets\split\self_test\predictions_20260705_112357.csv

Summary by Class:
  glass           5 images
  metal           2 images
  organic         1 images
  paper           2 images
  plastic         4 images

Average Confidence: 68.60%

D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python predict.py --folder datasets/split/test     
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 832 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
glass\trashnet_10.jpg                                        metal        68.66%
glass\trashnet_11.jpg                                        glass        58.76%
glass\trashnet_121.jpg                                       glass        50.35%
glass\trashnet_124.jpg                                       glass        61.29%
glass\trashnet_125.jpg                                       glass        67.93%
glass\trashnet_140.jpg                                       glass        59.22%
glass\trashnet_17.jpg                                        glass        68.05%
glass\trashnet_176.jpg                                       plastic      47.10%
glass\trashnet_190.jpg                                       plastic      99.28%
glass\trashnet_191.jpg                                       glass        50.76%
glass\trashnet_196.jpg                                       glass        67.64%
glass\trashnet_208.jpg                                       glass        53.58%
glass\trashnet_213.jpg                                       glass        97.98%
glass\trashnet_215.jpg                                       glass        68.85%
glass\trashnet_216.jpg                                       metal        63.60%
glass\trashnet_237.jpg                                       glass        61.28%
glass\trashnet_258.jpg                                       glass        80.17%
glass\trashnet_267.jpg                                       glass        94.54%
glass\trashnet_270.jpg                                       glass        93.99%
glass\trashnet_271.jpg                                       glass        83.77%
glass\trashnet_279.jpg                                       metal        51.44%
glass\trashnet_298.jpg                                       glass        93.85%
glass\trashnet_3.jpg                                         metal        82.07%
glass\trashnet_304.jpg                                       glass        90.81%
glass\trashnet_311.jpg                                       glass        64.29%
glass\trashnet_315.jpg                                       glass        66.47%
glass\trashnet_321.jpg                                       metal        81.82%
glass\trashnet_332.jpg                                       metal        61.20%
glass\trashnet_334.jpg                                       glass        43.31%
glass\trashnet_342.jpg                                       metal        52.07%
glass\trashnet_347.jpg                                       glass        99.98%
glass\trashnet_353.jpg                                       glass        53.10%
glass\trashnet_354.jpg                                       glass        80.04%
glass\trashnet_357.jpg                                       glass        89.07%
glass\trashnet_36.jpg                                        metal        55.75%
glass\trashnet_361.jpg                                       plastic      77.80%
glass\trashnet_389.jpg                                       glass        74.55%
glass\trashnet_392.jpg                                       plastic      57.74%
glass\trashnet_413.jpg                                       glass        78.67%
glass\trashnet_419.jpg                                       metal        48.73%
glass\trashnet_433.jpg                                       glass        58.04%
glass\trashnet_436.jpg                                       plastic      37.95%
glass\trashnet_44.jpg                                        plastic      74.81%
glass\trashnet_443.jpg                                       plastic      40.00%
glass\trashnet_446.jpg                                       glass        63.96%
glass\trashnet_449.jpg                                       glass        58.08%
glass\trashnet_451.jpg                                       metal        63.20%
glass\trashnet_455.jpg                                       plastic      80.62%
glass\trashnet_456.jpg                                       metal        67.72%
glass\trashnet_461.jpg                                       glass        71.11%
glass\trashnet_470.jpg                                       glass        39.27%
glass\trashnet_475.jpg                                       glass        62.48%
glass\trashnet_476.jpg                                       glass        83.03%
glass\trashnet_490.jpg                                       metal        65.09%
glass\trashnet_52.jpg                                        glass        96.87%
glass\trashnet_59.jpg                                        glass        97.47%
glass\trashnet_60.jpg                                        glass        87.25%
glass\trashnet_70.jpg                                        glass        70.95%
glass\trashnet_94.jpg                                        glass        49.59%
glass\v2_1004.jpg                                            metal        61.86%
glass\v2_1006.jpg                                            glass        45.38%
glass\v2_101.jpg                                             organic      81.28%
glass\v2_1019.jpg                                            metal        78.66%
glass\v2_1033.jpg                                            glass        88.24%
glass\v2_1056.jpg                                            glass        91.14%
glass\v2_106.jpg                                             plastic      66.40%
glass\v2_1060.jpg                                            glass        85.51%
glass\v2_1073.jpg                                            glass        87.86%
glass\v2_111.jpg                                             metal        60.84%
glass\v2_1119.jpg                                            glass        81.31%
glass\v2_1129.jpg                                            metal        61.20%
glass\v2_1134.jpg                                            glass        56.44%
glass\v2_1136.jpg                                            glass        93.17%
glass\v2_1141.jpg                                            glass        62.82%
glass\v2_1146.jpg                                            glass        95.00%
glass\v2_1149.jpg                                            plastic      52.02%
glass\v2_1155.jpg                                            paper        46.83%
glass\v2_1171.jpg                                            glass        79.11%
glass\v2_1177.jpg                                            metal        96.28%
glass\v2_1204.jpg                                            glass        91.07%
glass\v2_1248.jpg                                            glass        55.26%
glass\v2_1255.jpg                                            glass        50.54%
glass\v2_128.jpg                                             glass        78.10%
glass\v2_1288.jpg                                            paper        75.20%
glass\v2_1289.jpg                                            glass        81.50%
glass\v2_1319.jpg                                            metal        53.56%
glass\v2_1324.jpg                                            glass        96.01%
glass\v2_1327.jpg                                            glass        98.76%
glass\v2_133.jpg                                             glass        88.61%
glass\v2_1336.jpg                                            paper        74.91%
glass\v2_134.jpg                                             glass        74.65%
glass\v2_1352.jpg                                            glass        78.86%
glass\v2_1355.jpg                                            glass        99.43%
glass\v2_1356.jpg                                            paper        45.66%
glass\v2_1359.jpg                                            glass        95.91%
glass\v2_1363.jpg                                            glass        79.44%
glass\v2_137.jpg                                             glass        99.65%
glass\v2_1373.jpg                                            glass        94.13%
glass\v2_1375.jpg                                            glass        67.83%
glass\v2_1387.jpg                                            glass        59.72%
glass\v2_1390.jpg                                            glass        95.09%
glass\v2_1391.jpg                                            glass        95.94%
glass\v2_1395.jpg                                            glass        69.87%
glass\v2_14.jpg                                              glass        80.74%
glass\v2_1403.jpg                                            glass        60.72%
glass\v2_1404.jpg                                            glass        96.03%
glass\v2_1423.jpg                                            glass        81.00%
glass\v2_1429.jpg                                            metal        84.24%
glass\v2_1434.jpg                                            glass        92.17%
glass\v2_1442.jpg                                            plastic      40.60%
glass\v2_1449.jpg                                            organic      71.40%
glass\v2_145.jpg                                             glass        87.99%
glass\v2_1456.jpg                                            plastic      45.28%
glass\v2_1460.jpg                                            glass        67.07%
glass\v2_1473.jpg                                            metal        51.90%
glass\v2_1515.jpg                                            glass        56.16%
glass\v2_1517.jpg                                            plastic      41.05%
glass\v2_1521.jpg                                            glass        81.32%
glass\v2_1522.jpg                                            glass        62.09%
glass\v2_1523.jpg                                            glass        91.48%
glass\v2_153.jpg                                             metal        87.54%
glass\v2_1530.jpg                                            glass        48.57%
glass\v2_1533.jpg                                            glass        82.79%
glass\v2_1535.jpg                                            glass        84.68%
glass\v2_1541.jpg                                            glass        86.76%
glass\v2_1543.jpg                                            glass        72.36%
glass\v2_1560.jpg                                            glass        68.68%
glass\v2_1561.jpg                                            metal        41.58%
glass\v2_1571.jpg                                            glass        99.99%
glass\v2_1572.jpg                                            plastic      92.78%
glass\v2_1573.jpg                                            glass        74.54%
glass\v2_1613.jpg                                            metal        88.07%
glass\v2_1627.jpg                                            glass        60.02%
glass\v2_1629.jpg                                            glass        100.00%
glass\v2_1647.jpg                                            glass        98.73%
glass\v2_1680.jpg                                            glass        99.44%
glass\v2_1696.jpg                                            metal        83.02%
glass\v2_1706.jpg                                            glass        98.57%
glass\v2_171.jpg                                             glass        93.91%
glass\v2_18.jpg                                              paper        78.04%
glass\v2_180.jpg                                             organic      90.20%
glass\v2_187.jpg                                             glass        92.89%
glass\v2_202.jpg                                             glass        64.72%
glass\v2_225.jpg                                             glass        89.66%
glass\v2_230.jpg                                             glass        81.57%
glass\v2_239.jpg                                             paper        56.60%
glass\v2_252.jpg                                             glass        53.45%
glass\v2_291.jpg                                             glass        89.18%
glass\v2_295.jpg                                             glass        81.15%
glass\v2_308.jpg                                             glass        72.80%
glass\v2_318.jpg                                             plastic      64.96%
glass\v2_319.jpg                                             glass        51.66%
glass\v2_322.jpg                                             plastic      54.46%
glass\v2_327.jpg                                             glass        64.00%
glass\v2_33.jpg                                              glass        66.00%
glass\v2_35.jpg                                              glass        99.09%
glass\v2_361.jpg                                             glass        87.94%
glass\v2_363.jpg                                             plastic      49.24%
glass\v2_374.jpg                                             metal        55.66%
glass\v2_380.jpg                                             glass        45.45%
glass\v2_384.jpg                                             glass        88.94%
glass\v2_386.jpg                                             glass        84.46%
glass\v2_392.jpg                                             glass        51.27%
glass\v2_403.jpg                                             glass        98.98%
glass\v2_443.jpg                                             paper        33.32%
glass\v2_456.jpg                                             glass        94.74%
glass\v2_457.jpg                                             glass        92.90%
glass\v2_460.jpg                                             plastic      72.38%
glass\v2_464.jpg                                             metal        52.33%
glass\v2_474.jpg                                             paper        48.45%
glass\v2_486.jpg                                             glass        99.37%
glass\v2_50.jpg                                              glass        86.23%
glass\v2_526.jpg                                             glass        41.32%
glass\v2_531.jpg                                             glass        60.18%
glass\v2_540.jpg                                             glass        97.47%
glass\v2_541.jpg                                             glass        49.70%
glass\v2_543.jpg                                             glass        91.46%
glass\v2_545.jpg                                             glass        78.46%
glass\v2_546.jpg                                             glass        59.22%
glass\v2_563.jpg                                             glass        40.18%
glass\v2_564.jpg                                             glass        85.48%
glass\v2_573.jpg                                             glass        81.45%
glass\v2_574.jpg                                             glass        99.48%
glass\v2_588.jpg                                             glass        98.34%
glass\v2_597.jpg                                             glass        57.74%
glass\v2_60.jpg                                              glass        98.21%
glass\v2_640.jpg                                             glass        90.06%
glass\v2_65.jpg                                              glass        50.97%
glass\v2_657.jpg                                             plastic      74.81%
glass\v2_658.jpg                                             plastic      77.67%
glass\v2_677.jpg                                             glass        88.94%
glass\v2_679.jpg                                             metal        51.65%
glass\v2_687.jpg                                             metal        95.02%
glass\v2_69.jpg                                              organic      64.44%
glass\v2_693.jpg                                             plastic      46.32%
glass\v2_706.jpg                                             organic      76.20%
glass\v2_710.jpg                                             glass        79.56%
glass\v2_728.jpg                                             glass        97.77%
glass\v2_729.jpg                                             glass        44.45%
glass\v2_737.jpg                                             glass        74.69%
glass\v2_771.jpg                                             glass        72.40%
glass\v2_775.jpg                                             plastic      39.06%
glass\v2_777.jpg                                             glass        58.01%
glass\v2_787.jpg                                             glass        55.34%
glass\v2_804.jpg                                             glass        96.86%
glass\v2_823.jpg                                             glass        99.85%
glass\v2_825.jpg                                             paper        56.07%
glass\v2_830.jpg                                             glass        94.26%
glass\v2_848.jpg                                             glass        36.59%
glass\v2_851.jpg                                             glass        76.23%
glass\v2_857.jpg                                             glass        90.85%
glass\v2_864.jpg                                             glass        93.98%
glass\v2_891.jpg                                             paper        59.79%
glass\v2_91.jpg                                              glass        85.36%
glass\v2_929.jpg                                             glass        58.42%
glass\v2_961.jpg                                             glass        97.27%
glass\v2_962.jpg                                             plastic      82.56%
glass\v2_963.jpg                                             glass        35.97%
glass\v2_970.jpg                                             glass        97.95%
glass\v2_976.jpg                                             glass        85.18%
glass\v2_977.jpg                                             metal        58.46%
glass\v2_982.jpg                                             glass        65.49%
glass\v2_984.jpg                                             glass        77.26%
glass\v2_994.jpg                                             glass        57.35%
glass\v2_999.jpg                                             plastic      60.30%
metal\trashnet_112.jpg                                       metal        97.16%
metal\trashnet_140.jpg                                       plastic      36.08%
metal\trashnet_150.jpg                                       metal        33.70%
metal\trashnet_154.jpg                                       metal        90.98%
metal\trashnet_17.jpg                                        metal        98.69%
metal\trashnet_172.jpg                                       metal        92.23%
metal\trashnet_181.jpg                                       metal        99.37%
metal\trashnet_182.jpg                                       metal        98.86%
metal\trashnet_188.jpg                                       glass        44.75%
metal\trashnet_191.jpg                                       metal        98.97%
metal\trashnet_192.jpg                                       metal        55.88%
metal\trashnet_208.jpg                                       metal        99.78%
metal\trashnet_232.jpg                                       metal        94.18%
metal\trashnet_233.jpg                                       metal        87.55%
metal\trashnet_248.jpg                                       metal        29.03%
metal\trashnet_264.jpg                                       metal        96.63%
metal\trashnet_277.jpg                                       metal        89.93%
metal\trashnet_294.jpg                                       metal        94.65%
metal\trashnet_297.jpg                                       metal        47.92%
metal\trashnet_317.jpg                                       metal        67.98%
metal\trashnet_323.jpg                                       metal        98.53%
metal\trashnet_335.jpg                                       metal        87.65%
metal\trashnet_339.jpg                                       metal        56.87%
metal\trashnet_345.jpg                                       metal        94.29%
metal\trashnet_350.jpg                                       metal        50.21%
metal\trashnet_358.jpg                                       metal        97.30%
metal\trashnet_363.jpg                                       metal        50.24%
metal\trashnet_364.jpg                                       metal        99.42%
metal\trashnet_373.jpg                                       metal        51.94%
metal\trashnet_39.jpg                                        metal        50.50%
metal\trashnet_392.jpg                                       metal        76.93%
metal\trashnet_393.jpg                                       metal        84.82%
metal\trashnet_8.jpg                                         metal        84.86%
metal\trashnet_89.jpg                                        plastic      60.97%
metal\trashnet_97.jpg                                        metal        68.61%
metal\v2_104.jpg                                             metal        70.82%
metal\v2_117.jpg                                             metal        95.19%
metal\v2_125.jpg                                             metal        54.28%
metal\v2_173.jpg                                             paper        94.04%
metal\v2_175.jpg                                             metal        42.96%
metal\v2_180.jpg                                             metal        40.01%
metal\v2_186.jpg                                             metal        94.74%
metal\v2_187.jpg                                             glass        43.87%
metal\v2_216.jpg                                             metal        90.39%
metal\v2_217.jpg                                             plastic      36.08%
metal\v2_221.jpg                                             paper        61.59%
metal\v2_244.jpg                                             metal        36.89%
metal\v2_248.jpg                                             metal        74.48%
metal\v2_256.jpg                                             metal        99.94%
metal\v2_261.jpg                                             paper        48.81%
metal\v2_263.jpg                                             glass        47.35%
metal\v2_264.jpg                                             metal        48.64%
metal\v2_278.jpg                                             plastic      85.21%
metal\v2_286.jpg                                             metal        98.86%
metal\v2_298.jpg                                             metal        98.99%
metal\v2_305.jpg                                             metal        96.76%
metal\v2_317.jpg                                             glass        59.68%
metal\v2_332.jpg                                             metal        99.51%
metal\v2_343.jpg                                             metal        42.90%
metal\v2_354.jpg                                             metal        82.10%
metal\v2_356.jpg                                             metal        38.30%
metal\v2_366.jpg                                             paper        66.25%
metal\v2_38.jpg                                              metal        70.65%
metal\v2_398.jpg                                             metal        99.16%
metal\v2_402.jpg                                             metal        95.80%
metal\v2_403.jpg                                             metal        77.79%
metal\v2_411.jpg                                             metal        81.45%
metal\v2_417.jpg                                             metal        99.20%
metal\v2_437.jpg                                             metal        90.86%
metal\v2_441.jpg                                             paper        73.30%
metal\v2_454.jpg                                             metal        80.58%
metal\v2_455.jpg                                             metal        58.33%
metal\v2_464.jpg                                             glass        34.34%
metal\v2_468.jpg                                             metal        76.20%
metal\v2_47.jpg                                              metal        98.36%
metal\v2_472.jpg                                             metal        32.19%
metal\v2_478.jpg                                             metal        84.80%
metal\v2_496.jpg                                             metal        95.14%
metal\v2_498.jpg                                             metal        41.96%
metal\v2_50.jpg                                              metal        83.47%
metal\v2_523.jpg                                             glass        51.41%
metal\v2_54.jpg                                              metal        91.21%
metal\v2_558.jpg                                             metal        99.78%
metal\v2_56.jpg                                              metal        47.11%
metal\v2_565.jpg                                             metal        57.55%
metal\v2_571.jpg                                             metal        96.08%
metal\v2_572.jpg                                             metal        74.78%
metal\v2_586.jpg                                             metal        64.99%
metal\v2_59.jpg                                              metal        74.83%
metal\v2_590.jpg                                             metal        92.80%
metal\v2_597.jpg                                             metal        71.57%
metal\v2_602.jpg                                             plastic      55.03%
metal\v2_62.jpg                                              metal        99.50%
metal\v2_621.jpg                                             metal        86.13%
metal\v2_641.jpg                                             metal        84.61%
metal\v2_646.jpg                                             metal        94.13%
metal\v2_647.jpg                                             metal        80.17%
metal\v2_648.jpg                                             metal        99.09%
metal\v2_649.jpg                                             paper        88.90%
metal\v2_651.jpg                                             paper        88.10%
metal\v2_659.jpg                                             metal        80.06%
metal\v2_661.jpg                                             paper        43.12%
metal\v2_668.jpg                                             metal        95.85%
metal\v2_679.jpg                                             metal        53.19%
metal\v2_68.jpg                                              metal        69.38%
metal\v2_687.jpg                                             metal        93.22%
metal\v2_719.jpg                                             paper        42.85%
metal\v2_735.jpg                                             metal        95.28%
metal\v2_743.jpg                                             metal        95.36%
metal\v2_75.jpg                                              plastic      70.21%
metal\v2_758.jpg                                             metal        96.31%
metal\v2_765.jpg                                             metal        94.65%
metal\v2_780.jpg                                             metal        92.33%
metal\v2_784.jpg                                             metal        89.63%
metal\v2_793.jpg                                             metal        92.49%
metal\v2_795.jpg                                             metal        59.93%
metal\v2_804.jpg                                             metal        87.53%
metal\v2_830.jpg                                             metal        96.43%
metal\v2_848.jpg                                             metal        70.11%
metal\v2_849.jpg                                             paper        61.08%
metal\v2_851.jpg                                             metal        53.59%
metal\v2_864.jpg                                             metal        34.37%
metal\v2_869.jpg                                             paper        99.82%
metal\v2_886.jpg                                             metal        92.61%
metal\v2_888.jpg                                             metal        71.90%
metal\v2_889.jpg                                             glass        35.89%
metal\v2_898.jpg                                             glass        51.42%
metal\v2_899.jpg                                             organic      88.67%
metal\v2_902.jpg                                             metal        85.40%
metal\v2_903.jpg                                             metal        99.33%
metal\v2_907.jpg                                             metal        93.83%
metal\v2_911.jpg                                             metal        92.15%
metal\v2_912.jpg                                             metal        45.54%
metal\v2_93.jpg                                              plastic      71.08%
organic\v2_109.jpg                                           organic      93.03%
organic\v2_110.jpg                                           organic      99.14%
organic\v2_113.jpg                                           organic      79.89%
organic\v2_122.jpg                                           organic      73.87%
organic\v2_134.jpg                                           organic      99.78%
organic\v2_148.jpg                                           organic      98.25%
organic\v2_153.jpg                                           organic      85.81%
organic\v2_160.jpg                                           organic      55.94%
organic\v2_163.jpg                                           organic      92.40%
organic\v2_165.jpg                                           organic      98.67%
organic\v2_176.jpg                                           organic      99.91%
organic\v2_178.jpg                                           organic      98.56%
organic\v2_190.jpg                                           organic      98.74%
organic\v2_201.jpg                                           glass        67.68%
organic\v2_211.jpg                                           organic      99.96%
organic\v2_22.jpg                                            organic      99.99%
organic\v2_227.jpg                                           organic      92.79%
organic\v2_238.jpg                                           organic      93.12%
organic\v2_257.jpg                                           organic      97.98%
organic\v2_272.jpg                                           organic      94.95%
organic\v2_288.jpg                                           organic      95.81%
organic\v2_297.jpg                                           organic      98.27%
organic\v2_299.jpg                                           organic      91.72%
organic\v2_319.jpg                                           organic      99.92%
organic\v2_32.jpg                                            organic      99.85%
organic\v2_326.jpg                                           organic      99.98%
organic\v2_327.jpg                                           organic      99.26%
organic\v2_34.jpg                                            organic      99.98%
organic\v2_353.jpg                                           organic      92.27%
organic\v2_360.jpg                                           organic      99.47%
organic\v2_362.jpg                                           organic      98.59%
organic\v2_370.jpg                                           organic      99.55%
organic\v2_376.jpg                                           organic      99.18%
organic\v2_389.jpg                                           organic      99.09%
organic\v2_390.jpg                                           organic      97.28%
organic\v2_421.jpg                                           organic      99.82%
organic\v2_430.jpg                                           paper        73.80%
organic\v2_435.jpg                                           metal        37.35%
organic\v2_452.jpg                                           organic      99.87%
organic\v2_454.jpg                                           plastic      41.55%
organic\v2_455.jpg                                           organic      99.95%
organic\v2_46.jpg                                            organic      99.97%
organic\v2_465.jpg                                           organic      99.84%
organic\v2_478.jpg                                           organic      99.66%
organic\v2_486.jpg                                           organic      99.97%
organic\v2_494.jpg                                           organic      95.43%
organic\v2_506.jpg                                           organic      94.49%
organic\v2_532.jpg                                           organic      99.62%
organic\v2_554.jpg                                           organic      86.53%
organic\v2_559.jpg                                           organic      99.96%
organic\v2_560.jpg                                           organic      49.46%
organic\v2_565.jpg                                           glass        70.21%
organic\v2_568.jpg                                           organic      99.69%
organic\v2_576.jpg                                           organic      96.79%
organic\v2_585.jpg                                           organic      94.35%
organic\v2_601.jpg                                           organic      98.84%
organic\v2_608.jpg                                           organic      92.17%
organic\v2_622.jpg                                           organic      91.58%
organic\v2_64.jpg                                            organic      97.62%
organic\v2_641.jpg                                           organic      99.84%
organic\v2_646.jpg                                           organic      99.95%
organic\v2_651.jpg                                           organic      96.19%
organic\v2_652.jpg                                           organic      98.93%
organic\v2_658.jpg                                           organic      99.91%
organic\v2_659.jpg                                           organic      79.35%
organic\v2_666.jpg                                           organic      88.12%
organic\v2_678.jpg                                           organic      89.54%
organic\v2_687.jpg                                           organic      94.05%
organic\v2_693.jpg                                           organic      99.97%
organic\v2_694.jpg                                           organic      97.15%
organic\v2_91.jpg                                            organic      99.95%
paper\trashnet_113.jpg                                       plastic      48.02%
paper\trashnet_114.jpg                                       glass        35.92%
paper\trashnet_135.jpg                                       paper        92.71%
paper\trashnet_149.jpg                                       paper        98.62%
paper\trashnet_151.jpg                                       paper        59.53%
paper\trashnet_154.jpg                                       paper        86.80%
paper\trashnet_159.jpg                                       paper        68.92%
paper\trashnet_167.jpg                                       paper        60.17%
paper\trashnet_17.jpg                                        paper        100.00%
paper\trashnet_189.jpg                                       paper        88.88%
paper\trashnet_19.jpg                                        paper        72.69%
paper\trashnet_191.jpg                                       paper        97.20%
paper\trashnet_201.jpg                                       paper        95.82%
paper\trashnet_204.jpg                                       paper        97.32%
paper\trashnet_219.jpg                                       paper        99.70%
paper\trashnet_262.jpg                                       paper        99.99%
paper\trashnet_265.jpg                                       paper        100.00%
paper\trashnet_28.jpg                                        paper        80.70%
paper\trashnet_283.jpg                                       plastic      43.64%
paper\trashnet_284.jpg                                       paper        77.46%
paper\trashnet_293.jpg                                       paper        89.33%
paper\trashnet_305.jpg                                       paper        75.09%
paper\trashnet_326.jpg                                       paper        97.89%
paper\trashnet_341.jpg                                       metal        49.49%
paper\trashnet_346.jpg                                       paper        97.81%
paper\trashnet_348.jpg                                       paper        86.41%
paper\trashnet_363.jpg                                       plastic      65.22%
paper\trashnet_371.jpg                                       paper        83.57%
paper\trashnet_374.jpg                                       paper        99.93%
paper\trashnet_391.jpg                                       paper        99.44%
paper\trashnet_399.jpg                                       paper        62.26%
paper\trashnet_4.jpg                                         metal        48.67%
paper\trashnet_415.jpg                                       paper        99.61%
paper\trashnet_417.jpg                                       plastic      83.32%
paper\trashnet_418.jpg                                       paper        96.96%
paper\trashnet_425.jpg                                       paper        99.09%
paper\trashnet_444.jpg                                       paper        98.17%
paper\trashnet_453.jpg                                       plastic      55.39%
paper\trashnet_465.jpg                                       paper        99.28%
paper\trashnet_469.jpg                                       glass        50.93%
paper\trashnet_48.jpg                                        paper        60.62%
paper\trashnet_488.jpg                                       paper        59.93%
paper\trashnet_493.jpg                                       paper        43.13%
paper\trashnet_506.jpg                                       paper        54.22%
paper\trashnet_510.jpg                                       paper        99.31%
paper\trashnet_513.jpg                                       paper        99.64%
paper\trashnet_518.jpg                                       paper        99.82%
paper\trashnet_536.jpg                                       paper        49.16%
paper\trashnet_544.jpg                                       paper        62.30%
paper\trashnet_558.jpg                                       plastic      67.67%
paper\trashnet_563.jpg                                       paper        52.03%
paper\trashnet_575.jpg                                       paper        99.82%
paper\trashnet_585.jpg                                       paper        95.93%
paper\trashnet_6.jpg                                         paper        99.96%
paper\trashnet_60.jpg                                        paper        99.91%
paper\trashnet_7.jpg                                         paper        99.75%
paper\trashnet_71.jpg                                        paper        99.60%
paper\trashnet_72.jpg                                        paper        49.54%
paper\trashnet_81.jpg                                        paper        44.43%
paper\trashnet_9.jpg                                         paper        61.13%
paper\trashnet_91.jpg                                        paper        66.58%
paper\v2_1016.jpg                                            paper        82.48%
paper\v2_1020.jpg                                            paper        53.11%
paper\v2_1025.jpg                                            paper        69.02%
paper\v2_1035.jpg                                            paper        69.22%
paper\v2_1036.jpg                                            paper        99.71%
paper\v2_1054.jpg                                            paper        98.91%
paper\v2_1073.jpg                                            glass        31.14%
paper\v2_1085.jpg                                            paper        99.70%
paper\v2_1097.jpg                                            paper        74.62%
paper\v2_1098.jpg                                            plastic      65.04%
paper\v2_1107.jpg                                            paper        48.43%
paper\v2_1127.jpg                                            paper        99.97%
paper\v2_1141.jpg                                            paper        99.55%
paper\v2_1149.jpg                                            plastic      50.11%
paper\v2_1151.jpg                                            paper        99.91%
paper\v2_1157.jpg                                            paper        99.97%
paper\v2_1169.jpg                                            paper        99.99%
paper\v2_1175.jpg                                            plastic      46.86%
paper\v2_1192.jpg                                            paper        91.86%
paper\v2_1194.jpg                                            paper        98.71%
paper\v2_1202.jpg                                            paper        79.32%
paper\v2_1205.jpg                                            paper        99.64%
paper\v2_1206.jpg                                            paper        90.73%
paper\v2_1212.jpg                                            paper        44.37%
paper\v2_1217.jpg                                            paper        57.44%
paper\v2_122.jpg                                             paper        99.96%
paper\v2_1226.jpg                                            paper        97.87%
paper\v2_1230.jpg                                            paper        99.87%
paper\v2_1243.jpg                                            glass        40.78%
paper\v2_1277.jpg                                            paper        99.15%
paper\v2_128.jpg                                             paper        99.63%
paper\v2_1285.jpg                                            paper        54.14%
paper\v2_1303.jpg                                            paper        85.09%
paper\v2_1318.jpg                                            paper        91.83%
paper\v2_1334.jpg                                            paper        53.40%
paper\v2_134.jpg                                             paper        95.22%
paper\v2_135.jpg                                             paper        99.99%
paper\v2_137.jpg                                             paper        68.54%
paper\v2_154.jpg                                             paper        85.11%
paper\v2_156.jpg                                             paper        80.72%
paper\v2_162.jpg                                             paper        84.59%
paper\v2_167.jpg                                             paper        97.34%
paper\v2_192.jpg                                             paper        90.02%
paper\v2_195.jpg                                             paper        99.94%
paper\v2_207.jpg                                             paper        93.78%
paper\v2_210.jpg                                             paper        99.95%
paper\v2_212.jpg                                             paper        39.20%
paper\v2_227.jpg                                             paper        99.45%
paper\v2_248.jpg                                             paper        63.78%
paper\v2_278.jpg                                             paper        93.87%
paper\v2_299.jpg                                             paper        99.80%
paper\v2_305.jpg                                             paper        99.91%
paper\v2_309.jpg                                             paper        98.14%
paper\v2_315.jpg                                             paper        99.55%
paper\v2_335.jpg                                             paper        63.16%
paper\v2_342.jpg                                             paper        99.82%
paper\v2_345.jpg                                             paper        98.88%
paper\v2_356.jpg                                             paper        83.25%
paper\v2_37.jpg                                              paper        73.63%
paper\v2_376.jpg                                             paper        90.08%
paper\v2_382.jpg                                             paper        95.59%
paper\v2_383.jpg                                             paper        98.88%
paper\v2_384.jpg                                             metal        56.90%
paper\v2_400.jpg                                             paper        99.72%
paper\v2_432.jpg                                             plastic      53.79%
paper\v2_446.jpg                                             paper        86.75%
paper\v2_451.jpg                                             paper        99.47%
paper\v2_454.jpg                                             metal        48.40%
paper\v2_462.jpg                                             paper        96.12%
paper\v2_469.jpg                                             metal        40.09%
paper\v2_51.jpg                                              paper        95.66%
paper\v2_516.jpg                                             plastic      45.27%
paper\v2_517.jpg                                             plastic      78.55%
paper\v2_519.jpg                                             paper        96.14%
paper\v2_523.jpg                                             paper        98.75%
paper\v2_530.jpg                                             paper        99.85%
paper\v2_534.jpg                                             paper        85.60%
paper\v2_54.jpg                                              paper        48.76%
paper\v2_544.jpg                                             plastic      76.55%
paper\v2_560.jpg                                             paper        99.13%
paper\v2_569.jpg                                             paper        99.43%
paper\v2_57.jpg                                              paper        99.53%
paper\v2_575.jpg                                             plastic      39.98%
paper\v2_604.jpg                                             paper        97.41%
paper\v2_627.jpg                                             paper        99.16%
paper\v2_632.jpg                                             paper        99.88%
paper\v2_636.jpg                                             paper        46.22%
paper\v2_641.jpg                                             plastic      81.68%
paper\v2_66.jpg                                              paper        97.82%
paper\v2_661.jpg                                             paper        51.24%
paper\v2_676.jpg                                             paper        98.60%
paper\v2_689.jpg                                             paper        63.31%
paper\v2_699.jpg                                             paper        81.82%
paper\v2_7.jpg                                               paper        99.95%
paper\v2_70.jpg                                              paper        95.07%
paper\v2_71.jpg                                              paper        99.99%
paper\v2_72.jpg                                              paper        93.07%
paper\v2_724.jpg                                             paper        99.98%
paper\v2_738.jpg                                             paper        60.60%
paper\v2_745.jpg                                             paper        77.96%
paper\v2_769.jpg                                             paper        83.58%
paper\v2_775.jpg                                             paper        99.75%
paper\v2_78.jpg                                              paper        38.91%
paper\v2_780.jpg                                             plastic      60.67%
paper\v2_784.jpg                                             paper        99.86%
paper\v2_793.jpg                                             paper        97.60%
paper\v2_797.jpg                                             paper        86.41%
paper\v2_804.jpg                                             paper        99.57%
paper\v2_805.jpg                                             paper        99.96%
paper\v2_815.jpg                                             paper        63.81%
paper\v2_830.jpg                                             paper        74.47%
paper\v2_835.jpg                                             paper        50.86%
paper\v2_841.jpg                                             metal        56.94%
paper\v2_843.jpg                                             paper        98.66%
paper\v2_852.jpg                                             organic      95.23%
paper\v2_860.jpg                                             paper        74.04%
paper\v2_863.jpg                                             plastic      62.76%
paper\v2_866.jpg                                             paper        88.46%
paper\v2_875.jpg                                             paper        60.31%
paper\v2_88.jpg                                              paper        63.09%
paper\v2_882.jpg                                             paper        99.47%
paper\v2_884.jpg                                             plastic      40.38%
paper\v2_900.jpg                                             metal        58.22%
paper\v2_906.jpg                                             paper        99.24%
paper\v2_915.jpg                                             paper        63.84%
paper\v2_931.jpg                                             plastic      65.90%
paper\v2_951.jpg                                             paper        92.24%
paper\v2_952.jpg                                             paper        97.31%
paper\v2_960.jpg                                             paper        50.06%
paper\v2_967.jpg                                             organic      93.76%
paper\v2_975.jpg                                             paper        99.36%
paper\v2_984.jpg                                             paper        99.75%
plastic\trashnet_0.jpg                                       metal        60.66%
plastic\trashnet_10.jpg                                      plastic      93.36%
plastic\trashnet_11.jpg                                      plastic      88.20%
plastic\trashnet_121.jpg                                     plastic      54.99%
plastic\trashnet_128.jpg                                     plastic      81.80%
plastic\trashnet_156.jpg                                     plastic      86.16%
plastic\trashnet_157.jpg                                     plastic      97.84%
plastic\trashnet_161.jpg                                     plastic      81.55%
plastic\trashnet_167.jpg                                     plastic      51.39%
plastic\trashnet_173.jpg                                     metal        71.97%
plastic\trashnet_177.jpg                                     plastic      60.51%
plastic\trashnet_179.jpg                                     plastic      57.40%
plastic\trashnet_189.jpg                                     paper        57.98%
plastic\trashnet_205.jpg                                     plastic      91.02%
plastic\trashnet_21.jpg                                      plastic      75.46%
plastic\trashnet_214.jpg                                     plastic      97.44%
plastic\trashnet_220.jpg                                     plastic      93.82%
plastic\trashnet_228.jpg                                     plastic      98.76%
plastic\trashnet_229.jpg                                     paper        54.98%
plastic\trashnet_241.jpg                                     plastic      95.57%
plastic\trashnet_243.jpg                                     plastic      99.35%
plastic\trashnet_245.jpg                                     plastic      98.74%
plastic\trashnet_276.jpg                                     plastic      68.13%
plastic\trashnet_277.jpg                                     plastic      99.11%
plastic\trashnet_280.jpg                                     plastic      93.26%
plastic\trashnet_285.jpg                                     plastic      67.74%
plastic\trashnet_290.jpg                                     plastic      83.78%
plastic\trashnet_293.jpg                                     plastic      66.10%
plastic\trashnet_296.jpg                                     plastic      95.91%
plastic\trashnet_301.jpg                                     plastic      59.44%
plastic\trashnet_337.jpg                                     plastic      76.33%
plastic\trashnet_347.jpg                                     plastic      98.43%
plastic\trashnet_352.jpg                                     glass        83.36%
plastic\trashnet_382.jpg                                     glass        48.56%
plastic\trashnet_39.jpg                                      plastic      79.76%
plastic\trashnet_405.jpg                                     organic      64.45%
plastic\trashnet_427.jpg                                     plastic      81.12%
plastic\trashnet_428.jpg                                     plastic      86.37%
plastic\trashnet_430.jpg                                     plastic      93.26%
plastic\trashnet_448.jpg                                     plastic      52.52%
plastic\trashnet_451.jpg                                     plastic      97.01%
plastic\trashnet_454.jpg                                     plastic      35.46%
plastic\trashnet_470.jpg                                     plastic      98.58%
plastic\trashnet_5.jpg                                       plastic      73.24%
plastic\trashnet_6.jpg                                       plastic      75.02%
plastic\trashnet_63.jpg                                      plastic      41.98%
plastic\trashnet_72.jpg                                      metal        46.18%
plastic\trashnet_91.jpg                                      plastic      84.53%
plastic\v2_1006.jpg                                          plastic      51.53%
plastic\v2_1021.jpg                                          plastic      85.05%
plastic\v2_1031.jpg                                          plastic      99.18%
plastic\v2_1035.jpg                                          plastic      98.71%
plastic\v2_1053.jpg                                          plastic      67.46%
plastic\v2_106.jpg                                           plastic      86.99%
plastic\v2_1085.jpg                                          plastic      99.76%
plastic\v2_1088.jpg                                          plastic      94.03%
plastic\v2_1117.jpg                                          plastic      82.74%
plastic\v2_112.jpg                                           plastic      69.28%
plastic\v2_1123.jpg                                          plastic      72.47%
plastic\v2_1134.jpg                                          plastic      75.24%
plastic\v2_1144.jpg                                          plastic      60.61%
plastic\v2_1154.jpg                                          organic      29.53%
plastic\v2_1157.jpg                                          plastic      54.72%
plastic\v2_1158.jpg                                          plastic      85.93%
plastic\v2_116.jpg                                           plastic      77.67%
plastic\v2_1161.jpg                                          plastic      83.70%
plastic\v2_1186.jpg                                          metal        71.97%
plastic\v2_1212.jpg                                          glass        54.40%
plastic\v2_1223.jpg                                          plastic      75.66%
plastic\v2_1227.jpg                                          plastic      99.05%
plastic\v2_1229.jpg                                          paper        38.51%
plastic\v2_1239.jpg                                          organic      28.74%
plastic\v2_1243.jpg                                          plastic      98.15%
plastic\v2_1247.jpg                                          plastic      28.27%
plastic\v2_1249.jpg                                          organic      48.22%
plastic\v2_1251.jpg                                          glass        58.46%
plastic\v2_1255.jpg                                          plastic      40.58%
plastic\v2_1263.jpg                                          plastic      84.86%
plastic\v2_1274.jpg                                          plastic      63.53%
plastic\v2_1275.jpg                                          plastic      40.58%
plastic\v2_1278.jpg                                          plastic      74.72%
plastic\v2_128.jpg                                           glass        81.71%
plastic\v2_1296.jpg                                          plastic      51.42%
plastic\v2_1303.jpg                                          plastic      75.32%
plastic\v2_1319.jpg                                          plastic      89.01%
plastic\v2_1323.jpg                                          plastic      73.37%
plastic\v2_1329.jpg                                          plastic      99.80%
plastic\v2_133.jpg                                           plastic      37.89%
plastic\v2_1350.jpg                                          plastic      44.72%
plastic\v2_1357.jpg                                          metal        29.38%
plastic\v2_1380.jpg                                          plastic      93.38%
plastic\v2_1389.jpg                                          plastic      57.33%
plastic\v2_1398.jpg                                          plastic      96.62%
plastic\v2_1409.jpg                                          metal        52.52%
plastic\v2_1426.jpg                                          plastic      95.26%
plastic\v2_1444.jpg                                          plastic      85.97%
plastic\v2_1459.jpg                                          paper        61.79%
plastic\v2_1467.jpg                                          plastic      67.30%
plastic\v2_147.jpg                                           plastic      89.43%
plastic\v2_1480.jpg                                          plastic      79.96%
plastic\v2_1488.jpg                                          plastic      61.47%
plastic\v2_1494.jpg                                          glass        42.81%
plastic\v2_1518.jpg                                          plastic      92.70%
plastic\v2_1520.jpg                                          plastic      72.09%
plastic\v2_1528.jpg                                          plastic      92.45%
plastic\v2_153.jpg                                           plastic      71.21%
plastic\v2_1532.jpg                                          plastic      53.77%
plastic\v2_1533.jpg                                          plastic      78.46%
plastic\v2_1534.jpg                                          metal        61.47%
plastic\v2_1539.jpg                                          glass        70.22%
plastic\v2_1543.jpg                                          paper        67.69%
plastic\v2_1548.jpg                                          plastic      73.30%
plastic\v2_1557.jpg                                          plastic      99.43%
plastic\v2_157.jpg                                           plastic      88.54%
plastic\v2_1578.jpg                                          plastic      99.76%
plastic\v2_1579.jpg                                          plastic      94.05%
plastic\v2_1580.jpg                                          plastic      99.84%
plastic\v2_1592.jpg                                          plastic      98.12%
plastic\v2_1593.jpg                                          plastic      64.74%
plastic\v2_165.jpg                                           metal        96.73%
plastic\v2_178.jpg                                           plastic      50.93%
plastic\v2_18.jpg                                            plastic      60.90%
plastic\v2_19.jpg                                            glass        67.59%
plastic\v2_213.jpg                                           plastic      64.79%
plastic\v2_214.jpg                                           plastic      96.02%
plastic\v2_217.jpg                                           plastic      75.47%
plastic\v2_225.jpg                                           plastic      73.02%
plastic\v2_232.jpg                                           plastic      97.02%
plastic\v2_243.jpg                                           plastic      73.76%
plastic\v2_249.jpg                                           plastic      73.28%
plastic\v2_257.jpg                                           paper        95.06%
plastic\v2_264.jpg                                           plastic      53.88%
plastic\v2_277.jpg                                           plastic      93.58%
plastic\v2_28.jpg                                            plastic      99.47%
plastic\v2_294.jpg                                           plastic      48.57%
plastic\v2_303.jpg                                           plastic      63.40%
plastic\v2_304.jpg                                           plastic      53.18%
plastic\v2_308.jpg                                           plastic      55.46%
plastic\v2_31.jpg                                            plastic      95.30%
plastic\v2_32.jpg                                            plastic      87.79%
plastic\v2_335.jpg                                           plastic      65.18%
plastic\v2_338.jpg                                           paper        99.38%
plastic\v2_339.jpg                                           plastic      90.31%
plastic\v2_34.jpg                                            plastic      92.79%
plastic\v2_340.jpg                                           plastic      81.37%
plastic\v2_341.jpg                                           plastic      55.84%
plastic\v2_353.jpg                                           plastic      68.13%
plastic\v2_370.jpg                                           plastic      81.81%
plastic\v2_387.jpg                                           plastic      80.61%
plastic\v2_409.jpg                                           plastic      91.40%
plastic\v2_42.jpg                                            plastic      92.12%
plastic\v2_421.jpg                                           paper        55.61%
plastic\v2_432.jpg                                           plastic      56.46%
plastic\v2_448.jpg                                           plastic      89.75%
plastic\v2_450.jpg                                           plastic      68.35%
plastic\v2_456.jpg                                           plastic      43.96%
plastic\v2_464.jpg                                           plastic      89.75%
plastic\v2_465.jpg                                           plastic      38.64%
plastic\v2_466.jpg                                           organic      99.27%
plastic\v2_47.jpg                                            plastic      95.89%
plastic\v2_480.jpg                                           plastic      96.13%
plastic\v2_498.jpg                                           plastic      70.14%
plastic\v2_5.jpg                                             plastic      89.49%
plastic\v2_502.jpg                                           plastic      96.90%
plastic\v2_508.jpg                                           glass        61.11%
plastic\v2_52.jpg                                            glass        67.76%
plastic\v2_524.jpg                                           plastic      83.46%
plastic\v2_536.jpg                                           glass        41.47%
plastic\v2_539.jpg                                           plastic      70.59%
plastic\v2_54.jpg                                            plastic      86.87%
plastic\v2_588.jpg                                           plastic      87.94%
plastic\v2_591.jpg                                           plastic      99.33%
plastic\v2_603.jpg                                           plastic      92.47%
plastic\v2_637.jpg                                           plastic      63.37%
plastic\v2_642.jpg                                           plastic      83.04%
plastic\v2_645.jpg                                           plastic      97.97%
plastic\v2_65.jpg                                            glass        54.22%
plastic\v2_661.jpg                                           plastic      97.80%
plastic\v2_665.jpg                                           plastic      73.47%
plastic\v2_667.jpg                                           plastic      97.31%
plastic\v2_669.jpg                                           plastic      49.22%
plastic\v2_685.jpg                                           paper        63.19%
plastic\v2_737.jpg                                           plastic      96.72%
plastic\v2_740.jpg                                           paper        51.52%
plastic\v2_744.jpg                                           plastic      89.20%
plastic\v2_755.jpg                                           plastic      96.28%
plastic\v2_768.jpg                                           paper        54.98%
plastic\v2_769.jpg                                           plastic      49.17%
plastic\v2_784.jpg                                           plastic      86.12%
plastic\v2_791.jpg                                           plastic      89.72%
plastic\v2_800.jpg                                           plastic      89.15%
plastic\v2_825.jpg                                           paper        47.66%
plastic\v2_839.jpg                                           plastic      93.27%
plastic\v2_842.jpg                                           plastic      66.19%
plastic\v2_843.jpg                                           plastic      98.84%
plastic\v2_854.jpg                                           glass        41.72%
plastic\v2_856.jpg                                           paper        64.86%
plastic\v2_857.jpg                                           glass        61.66%
plastic\v2_869.jpg                                           plastic      71.08%
plastic\v2_872.jpg                                           plastic      84.98%
plastic\v2_902.jpg                                           glass        83.58%
plastic\v2_91.jpg                                            organic      94.19%
plastic\v2_916.jpg                                           plastic      84.53%
plastic\v2_934.jpg                                           plastic      70.70%
plastic\v2_959.jpg                                           plastic      96.27%
plastic\v2_962.jpg                                           plastic      80.48%
plastic\v2_977.jpg                                           plastic      76.81%
plastic\v2_982.jpg                                           plastic      48.70%
plastic\v2_984.jpg                                           plastic      80.93%
====================================================================================

✓ Processing complete! Total: 832 images
✓ Results saved to: datasets\split\test\predictions_20260705_113316.csv

Summary by Class:
  glass         186 images
  metal         152 images
  organic        80 images
  paper         196 images
  plastic       218 images

Average Confidence: 77.33%









D:\Projects_on_Python\Waste_Classifier_using_keras_seq_cnn\waste_classifier>python predict_test.py --folder datasets/split/train
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 5 class folders

Processing 6627 images...

[6627/6627] ✓ plastic: v2_999.jpgg.jpgg


======================================================================
Test Results on: train
======================================================================

Total Images Processed: 6627
Overall Accuracy: 0.8155 (5404/6627)

Class          Precision   Recall      F1-Score    Support   
----------------------------------------------------------------------
glass          0.8622      0.7418      0.7975      1789      
metal          0.7521      0.8293      0.7888      1072      
organic        0.8344      0.9463      0.8868      559       
paper          0.8366      0.8459      0.8412      1544      
plastic        0.7908      0.8136      0.8020      1663      
----------------------------------------------------------------------
Macro Avg      0.8152      0.8354      0.8233      
======================================================================

✓ Report saved to: outputs\test_report_train.txt
✓ Confusion matrix saved to: outputs\confusion_matrix_train.png
✓ Class metrics plot saved to: outputs\class_metrics_train.png

✓ Test completed successfully!