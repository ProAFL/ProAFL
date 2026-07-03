# Datactive

**This repository is an implementation of the paper: Datactive: Dataset Debugging for Object Detection Systems.**

Object detection models are seamlessly integrated into numerous intelligent software systems, playing a crucial role in various tasks. 
These models are typically constructed upon human-annotated datasets, whose quality may significantly influence the model's performance and reliability. 
Erroneous and inadequate annotated datasets can induce classification/localization inaccuracies during deployment, precipitating security breaches or traffic accidents that inflict property damage or even loss of life.
Therefore, ensuring and improving data quality is a crucial issue for the reliability of the object detection system.

In this paper, we propose *Datactive*, a data fault localization technique for object detection systems.
 *Datactive* is designed to locate various types of data faults including mis-localization and missing objects, without utilizing the prediction of object detection models trained on dirty datasets.
To achieve this, we first construct foreground-only and background-included datasets via data disassembling strategies, and then employ a robust learning method to train classifiers using these disassembled datasets. Based on the predictions made by these classifiers,  *Datactive* produces a unified suspiciousness score for both foreground annotations and image backgrounds.
It allows testers to easily identify and correct faulty or missing annotations with minimal effort.
To validate the effectiveness of our technique, we conducted experiments on three application datasets using six baselines, and demonstrated the superiority of  *Datactive* from various aspects.

We also explored *Datactive*'s ability to find natural data faults and its application in both training and evaluation scenarios.
![overview](./pictures/overview.Png) 

## Installation
`pip install -r requirements.txt`

## Usage
You should first transform the annotation of the object detection dataset into the following format `.json` file:
```python
[
    # an instance
    {
        "image_name": "000000312552.jpg",
        "image_size": [
            400,
            300
        ],
        "boxes": [ #x1,y1,x2,y2
            165.86,
            129.96,
            207.41000000000003,
            161.83
        ],
        "labels": 1,
        "image_id": 312552,
        "area": 798.7644499999996,
        "iscrowd": 0,
        "fault_type": 0
    },
  ...
]
```

Then you can run the following command to debug the dataset:
+ `python demo.py --dataset ./dataset/COCO --trainlabel ./dataset/COCO/trainlabel.json --testlabel ./dataset/COCO/testlabel.json --classnum 80`

Parameter explanation:

`--dataset` : Location of dataset image storage.

`--trainlabel` : The above transformed trainlabel.

`--testlabel` : The above transformed testlabel.

`--classnum` : Number of categories in the dataset.

## Results

*Datactive* improves the performance of the model：

| Map@0.5    | Method-    | Test Set |         |           |         |            |         |         |          |          |
|------------|------------|----------|---------|-----------|---------|------------|---------|---------|----------|----------|
|            | Model      | Original | Dirty   | Datactive | CE-Loss | Focal-Loss | Entropy | Margin  | DeepGini | Cleanlab |
| Pascal VOC | Original   | 0.8440   | 0.3980  | 0.7362    | 0.6427  | 0.6362     | 0.5180  | 0.5143  | 0.5148   | 0.6021   |
|            | Dirty      | 0.7796   | 0.3673  | 0.6794    | 0.5858  | 0.5800     | 0.4584  | 0.4564  | 0.4569   | 0.5659   |
|            | Datactive  | 0.8299   | 0.3933  | 0.7271    | 0.6314  | 0.6250     | 0.5099  | 0.5049  | 0.5066   | 0.5943   |
|            | CE-Loss    | 0.8214   | 0.3886  | 0.7163    | 0.6241  | 0.6184     | 0.5039  | 0.4995  | 0.5003   | 0.5880   |
|            | Focal-Loss | 0.8195   | 0.3883  | 0.7148    | 0.6251  | 0.6184     | 0.5026  | 0.4991  | 0.5002   | 0.5891   |
|            | Entropy    | 0.8205   | 0.3893  | 0.7192    | 0.6251  | 0.6186     | 0.5014  | 0.4971  | 0.4974   | 0.5893   |
|            | Margin     | 0.8225   | 0.3912  | 0.7208    | 0.6241  | 0.6180     | 0.5047  | 0.5007  | 0.5016   | 0.5899   |
|            | DeepGini   | 0.8274   | 0.3911  | 0.7230    | 0.6303  | 0.6237     | 0.5068  | 0.5032  | 0.5037   | 0.5891   |
|            | Cleanlab   | 0.7982   | 0.3737  | 0.6981    | 0.5995  | 0.5940     | 0.4808  | 0.4786  | 0.4788   | 0.5753   |
| VisDrone   | Original   | 0.2993   | 0.1342  | 0.2299    | 0.1729  | 0.1731     | 0.1956  | 0.1961  | 0.1960   | 0.1833   |
|            | Dirty      | 0.2607   | 0.1213  | 0.2025    | 0.1480  | 0.1483     | 0.1718  | 0.1729  | 0.1724   | 0.1626   |
|            | Datactive  | 0.2782   | 0.1267  | 0.2157    | 0.1613  | 0.1615     | 0.1840  | 0.1851  | 0.1845   | 0.1738   |
|            | CE-Loss    | 0.2643   | 0.1222  | 0.2045    | 0.1551  | 0.1553     | 0.1765  | 0.1779  | 0.1773   | 0.1665   |
|            | Focal-Loss | 0.2642   | 0.1206  | 0.2038    | 0.1540  | 0.1544     | 0.1755  | 0.1771  | 0.1763   | 0.1656   |
|            | Entropy    | 0.2773   | 0.1268  | 0.2143    | 0.1601  | 0.1602     | 0.1848  | 0.1857  | 0.1851   | 0.1729   |
|            | Margin     | 0.2749   | 0.1256  | 0.2123    | 0.1591  | 0.1592     | 0.1842  | 0.1852  | 0.1846   | 0.1725   |
|            | DeepGini   | 0.2777   | 0.1264  | 0.2145    | 0.1609  | 0.1610     | 0.1857  | 0.1863  | 0.1859   | 0.1731   |
|            | Cleanlab   | 0.2704   | 0.1237  | 0.2087    | 0.1567  | 0.1569     | 0.1793  | 0.1806  | 0.1798   | 0.1701   |
| KITTI      | Original   | 0.8827   | 0.3114  | 0.7936    | 0.5657  | 0.5515     | 0.4458  | 0.4424  | 0.4437   | 0.4907   |
|            | Dirty      | 0.7986   | 0.2901  | 0.7174    | 0.5081  | 0.4961     | 0.3869  | 0.3847  | 0.3866   | 0.4608   |
|            | Datactive  | 0.8701   | 0.3064  | 0.7819    | 0.5513  | 0.5399     | 0.4349  | 0.4329  | 0.4350   | 0.4855   |
|            | CE-Loss    | 0.8357   | 0.3007  | 0.7514    | 0.5382  | 0.5262     | 0.4167  | 0.4143  | 0.4163   | 0.4698   |
|            | Focal-Loss | 0.8280   | 0.2988  | 0.7445    | 0.5357  | 0.5231     | 0.4141  | 0.4123  | 0.4137   | 0.4677   |
|            | Entropy    | 0.8399   | 0.3056  | 0.7553    | 0.5442  | 0.5320     | 0.4315  | 0.4280  | 0.4301   | 0.4790   |
|            | Margin     | 0.8539   | 0.3063  | 0.7675    | 0.5503  | 0.5374     | 0.4321  | 0.4305  | 0.4320   | 0.4796   |
|            | DeepGini   | 0.8546   | 0.3086  | 0.7701    | 0.5488  | 0.5356     | 0.4328  | 0.4308  | 0.4323   | 0.4828   |
|            | Cleanlab   | 0.8265   | 0.2977  | 0.7424    | 0.5294  | 0.5168     | 0.4117  | 0.4099  | 0.4112   | 0.4690   |


Some `demo.py` results on COCO are shown below :
<div><table frame=void>	<!--用了<div>进行封装-->
	<tr>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/35.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td>    
     	<td><div><center>	<!--第二张图片-->
    		<img src="./pictures/67.png"
                 height="120"/>	
    		<br>
        </center></div></td>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/111.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/123.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/141.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
	</tr>
</table></div>
