# Assignment 5

- Date : 11/19/2025


## Q1. Classification Model (40 points)

- Report the test accuracy.
    - test accuracy: 0.9664

- Visualize a few random test point clouds and mention the predicted classes for each. Also visualize at least 1 failure prediction for each class (chair, vase and lamp),  and provide interpretation in a few sentences.  


| Case  | ![](data/data_0_gt_0.0_pred_0.gif)  | ![](data/data_700_gt_1_pred_1.gif)  | ![](data/data_800_gt_2_pred_2.gif)  |  ![](data/data_77_pts_10000_gt_0_pred_2.gif) | ![](data/data_652_gt_1_pred_2.gif)  | ![](data/data_922_gt_2_pred_1.gif)  |
|---|---|---|---|---|---|---|
| Ground truth  | chair  | vase  |  lamp | chair    |vase  | lamp  |
| Predicted Class  | chair  | vase  |  lamp |   lamp  |lamp |  vase |
| Correct/Failure  | correct  | correct  | correct  |  failure    |failure  | failure  |

Most of the failures occur on vases and lamps. In particular, vases containing flowers or plants, as well as lamps with morden designs, are more prone to misclassification. This is likely because vases and lamps often share elongated or rounded forms, whereas chairs typically have shapes that are clearly distinguishable from these objects.

## Q2. Segmentation Model (40 points) 

- Report the test accuracy.
    - test accuracy: 0.8984

- Visualize segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth, report the prediction accuracy for each object, and provide interpretation in a few sentences.

| Prediction  | ![](data/pred_seg_0_acc_0.9602.gif)  |  ![](data/pred_seg_360_acc_0.9678.gif) | ![](data/pred_seg_320_acc_0.9435.gif)  | ![](data/pred_seg_225_acc_0.5646.gif)  | ![](data/pred_seg_200_acc_0.7967.gif)   |
|---|---|---|---|---|---|
| Ground Truth  | ![](data/gt_seg_0_acc_0.9602.gif)  | ![](data/gt_seg_360_acc_0.9678.gif) |  ![](data/gt_seg_320_acc_0.9435.gif) | ![](data/gt_seg_225_acc_0.5646.gif)  | ![](data/gt_seg_200_acc_0.7967.gif)  |
| accuracy  | 0.9602  | 0.9678  | 0.9435  | 0.5646  | 0.7967  |
| prediction  | good  | good  | good  | bad  |  bad |


It appears that standard chair designs are more likely to be segmented correctly, while chairs with sofa-like shapes or components are more prone to segmentation errors.


## Q3. Robustness Analysis (20 points) 

Deliverables: On your website, for each experiment

- Describe your procedure 
- For each task, report test accuracy and visualization on a few samples, in comparison with your results from Q1 & Q2.


#### 1. Rotation

I rotate the input point clouds by 30 and 60 degrees. This is done by multiple the points with a rotation matrix. The roation matrix is implemented as following :
```
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ],dtype=np.float32)
```


- Overall Classification Test Accueacy : 
    - Before rotation : 0.9664
    - Rotate 30 degree : 0.8395
    - Rotate 60 degree : 0.3337

- Visualization of classification result
    - Case 0

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Visualization  |  ![](data/data_0_gt_0.0_pred_0.gif)   | ![](data/data_0_rot_30_gt_0_pred_0.gif)  | ![](data/data_0_rot_60_gt_0_pred_1.gif)  | 
    | Ground truth  | chair  | chair  |  chair |
    | Predicted Class  | chair  |  chair | vase  | 
    | Correct/Failure  | correct  | correct  | failure  | 

    - Case 620

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Visualization  |  ![](data/data_620_pts_10000_gt_1_pred_1.gif)   | ![](data/data_620_rot_30_gt_1_pred_2.gif)  | ![](data/data_620_rot_60_gt_1_pred_2.gif)  | 
    | Ground truth  | vase  | vase  |  vase |
    | Predicted Class  | vase  |  lamps | lamps  | 
    | Correct/Failure  | correct  | failure  | failure  | 

    - Case 780

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Visualization  |  ![](data/data_780_pts_10000_gt_2_pred_2.gif)   | ![](data/data_780_rot_30_gt_2_pred_1.gif)  | ![](data/data_780_rot_60_gt_2_pred_1.gif)  | 
    | Ground truth  | lamps  | lamps  |  lamps |
    | Predicted Class  | lamps  |  vase | vase  | 
    | Correct/Failure  | correct  | failure  | failure  | 


- Overall Segmentation Test Accueacy : 
    - Before rotation : 0.8984
    - Rotate 30 degree : 0.7242
    - Rotate 60 degree : 0.4544

- Visualization of Segmentation result
    - Case 0

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_0_pts_10000_acc_0.9602.gif)   | ![](data/pred_seg_0_rot_30_acc_0.793.gif)  | ![](data/pred_seg_0_rot_60_acc_0.4448.gif)  | 
    | Ground truth  | ![](data/gt_seg_0_acc_0.9602.gif)  | ![](data/gt_seg_0_rot_30_acc_0.793.gif)  | ![](data/gt_seg_0_rot_60_acc_0.4448.gif)  |
    | Accuracy | 0.9602  | 0.7930  | 0.4448  | 

    - Case 100

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_100_acc_0.9554.gif)   | ![](data/pred_seg_100_rot_30_acc_0.7889.gif)  | ![](data/pred_seg_100_rot_60_acc_0.5163.gif)  | 
    | Ground truth  | ![](data/gt_seg_100_acc_0.9554.gif)  | ![](data/gt_seg_100_rot_30_acc_0.7889.gif)  | ![](data/gt_seg_100_rot_60_acc_0.5163.gif)  | 
    | Accuracy | 0.9554  | 0.7889  | 0.5163  | 

    - Case 120

    | Rotation  | No Rotation| Rotate 30 degree | Rotate 60 degree |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_120_acc_0.9659.gif)   | ![](data/pred_seg_120_rot_30_acc_0.8296.gif)  | ![](data/pred_seg_120_rot_60_acc_0.3577.gif)  | 
    | Ground truth  |   ![](data/gt_seg_120_acc_0.9659.gif)   | ![](data/gt_seg_120_rot_30_acc_0.8296.gif)  | ![](data/gt_seg_120_rot_60_acc_0.3577.gif)  | 
    | Accuracy | 0.9659  |  0.8296 | 0.3577  | 


The test accuracy decreases for both the classification and segmentation tasks as the rotation angle increases. This indicates that our PointNet-based model implementation is not robust to rotational variation, likely because we did not include the transformation block used in the original paper.



#### 2. number of points per object 

I input a different number of points points per object. That is, I set the num_samples to 1000 and compared it with the Q1&Q2 settgins (10000 points). The following are the visualization results.


- Classification Test accueacy : 
    - 10000 points (Q1 settings) :0.9664
    - 1000 points : 0.9601
    - 100 points : 0.9297

- Visualization of classification result
    - Case 0

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Visualization  |  ![](data/data_0_gt_0.0_pred_0.gif)   | ![](data/data_0_pts_1000_gt_0_pred_0.gif)  | ![](data/data_0_pts_100_gt_0_pred_0.gif)|
    | Ground truth  | chair  | chair  |  chair |
    | Predicted Class  | chair  |  chair | chair  | 
    | Correct/Failure  | correct  | correct  | correct  | 

    - Case 620

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Visualization  |  ![](data/data_620_pts_10000_gt_1_pred_1.gif) |![](data/data_620_pts_1000_gt_1_pred_1.gif)  | ![](data/data_620_pts_100_gt_1_pred_2.gif) | 
    | Ground truth  | vase  | vase  |  vase |
    | Predicted Class  | vase  |  vase | lamps  | 
    | Correct/Failure  | correct  | correct  | failure  | 
    
    - Case 800

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Visualization  |  ![](data/data_800_gt_2_pred_2.gif)   | ![](data/data_800_pts_1000_gt_2_pred_2.gif)  | ![](data/data_800_pts_100_gt_2_pred_2.gif)  | 
    | Ground truth  | lamps  | lamps  |  lamps |
    | Predicted Class  | lamps  |  lamps | lamps  | 
    | Correct/Failure  | correct  | correct  | correct  | 

- Segmentation Test accueacy : 
    - 10000 points (Q2 settings) : 0.8984
    - 1000 points : 0.8933
    - 100 points : 0.8027

- Visualization of Segmentation result
    - Case 0

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_0_acc_0.9602.gif)   | ![](data/pred_seg_0_pts_1000_acc_0.957.gif)  | ![](data/pred_seg_0_pts_100_acc_0.87.gif)  | 
    | Ground truth  | ![](data/gt_seg_0_acc_0.9602.gif)   | ![](data/gt_seg_0_pts_1000_acc_0.957.gif)  | ![](data/gt_seg_0_pts_100_acc_0.88.gif)  | 
    | Accuracy | 0.9602  | 0.9570  | 0.8800  | 

    - Case 100

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_100_acc_0.9554.gif)   | ![](data/pred_seg_100_pts_1000_acc_0.96.gif)  | ![](data/pred_seg_100_pts_100_acc_0.84.gif)  | 
    | Ground truth  | ![](data/gt_seg_100_acc_0.9554.gif)  | ![](data/gt_seg_100_pts_1000_acc_0.96.gif)  | ![](data/gt_seg_100_pts_100_acc_0.84.gif)  | 
    | Accuracy | 0.9554  | 0.9600  | 0.8400  | 

    - Case 120

    | Number of Points  | 10000| 1000 | 100 |
    |---|---|---|---|
    | Prediction  |  ![](data/pred_seg_120_acc_0.9659.gif)   | ![](data/pred_seg_120_pts_1000_acc_0.966.gif)  | ![](data/pred_seg_120_pts_100_acc_0.91.gif)  | 
    | Ground truth  |   ![](data/gt_seg_120_acc_0.9659.gif)   | ![](data/gt_seg_120_pts_1000_acc_0.966.gif)  | ![](data/gt_seg_120_pts_100_acc_0.91.gif)  |
    | Accuracy | 0.9659  |  0.9660 | 0.9100 | 



The test accuracy for both the classification and segmentation tasks shows only minor degradation, indicating that our PointNet-based model is robust to variations in the number of points per object. Although some cases with 100 points exhibit more noticeable drops, these results also show high variance due to the sampling randomness introduced when using such a small point count.



## Q4. Bonus Question - Locality (20 points)
Incorporate certain kind of locality as covered in the lecture (e.g. implement PointNet++, DGCNN, Transformers (https://arxiv.org/pdf/2012.09164v2.pdf), etc).

Deliverables: On your website, 

- specify the model you have implemented
- for each task, report the test accuracy of your best model, in comparison with your results from Q1 & Q2
- visualize results in comparison to ones obtained in the earlier parts

N/A