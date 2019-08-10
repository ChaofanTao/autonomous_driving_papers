# Autonomous Driving Papers
Papers collection about autonomous driving

- **Latest Publications**
 <br/> 
 <br/> 
 <br/> 
 <br/> 
 <br/> 
  
  

---

- **Trajectory Prediction**
    - 19-ICCV-The Trajectron: Probabilistic Multi-Agent Trajectory Modeling with Dynamic Spatiotemporal Graphs, [pdf](https://arxiv.org/pdf/1810.05993.pdf)
    - 19-ICCV-Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction, [pdf](https://arxiv.org/abs/1907.10178)
    - 19-ICCV-PRECOG: PREdiction Conditioned On Goals in Visual Multi-Agent Settings, [pdf](https://arxiv.org/pdf/1905.01296.pdf), [project](https://sites.google.com/view/precog)
    - 18-CVPR-Social GAN Socially Acceptable Trajectories with GANs, [pdf](https://arxiv.org/pdf/1803.10892.pdf), [pytorch code](https://github.com/agrimgupta92/sgan)
        - 把trajectory prediction看成点列生成问题，所以用GAN-based architecture。
        - 在top k confident predictions用L2 loss, 增加variety。
    - 19-AAAI-TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents, [pdf](https://arxiv.org/pdf/1811.02146.pdf), [pytorch code](https://github.com/huang-xx/TrafficPredict)
        - 提出 instance layer来建模实例之间的运动和相互影响； 用category layer来建模同类别实例之间的相似性。
        - 用统一的模型预测汽车、人、自行车的轨迹。
    - 18-CVPR Trajnet Workshop -Convolutional Social Pooling for Vehicle Trajectory Prediction, [pdf](https://arxiv.org/abs/1805.06771)
        - Conv层+pooling层来代替social lstm里面的social pooling。方法是把lstm输出的hidden feature放在一个tensor里，tensor的列数是lane的个数，行数是输入总高度/单个car的长度，深度数是hidden dimension。
    - 16-CVPR-Social LSTM Human Trajectory Prediction in Crowded Spaces, [pdf](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf), [pytorch code](https://github.com/quancore/social-lstm)
        - 提出social pooling对空间上相近的人的关系建模，方法是在人工设置大小的grid里，把同一个grid的hidden feature累加。

---

- **3D Tracking/Object Detection/Recognition, etc**
    - 19-ICCV-Deep HoughVoting for 3D Object Detection in Point Clouds, [pdf](https://arxiv.org/abs/1904.09664)
    - 19-ICCV-Joint Monocular 3D Vehicle Detection and Tracking, [pdf](https://arxiv.org/abs/1811.10742), [pytorch code](https://github.com/ucbdrive/3d-vehicle-tracking)
    - 19-ICCV-oral-Deep Hough Voting for 3D Object Detection in Point Clouds, [pdf](https://arxiv.org/abs/1904.09664)
    - 19-ICCV-PU-GAN: a Point Cloud Upsampling Adversarial Network, [pdf](https://arxiv.org/pdf/1907.10844.pdf)
    - 19-CVPR-ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving, [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Song_ApolloCar3D_A_Large_3D_Car_Instance_Understanding_Benchmark_for_Autonomous_CVPR_2019_paper.pdf)
    - 19-ICCV-3D-RelNet: Joint Object and Relational Network for 3D Prediction, [pdf](https://arxiv.org/pdf/1906.02729.pdf), [pytorch code](https://nileshkulkarni.github.io/relative3d/)
    - 19-ICCV-3D Point Cloud Learning for Large-scale Environment Analysis and Place Recognition, [pdf](https://arxiv.org/pdf/1812.07050.pdf)
    - 19-ICCV-Exploring the Limitations of Behavior Cloning for Autonomous Driving [pdf](https://arxiv.org/pdf/1904.08980.pdf), [python code](https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md)
    - 19-ICCV-oral-Can GCNs Go as Deep as CNNs? [pdf](https://arxiv.org/pdf/1904.03751.pdf), [tensorflow code](https://github.com/lightaime/deep_gcns)
        - 把CNN的residual/dense connection和dilated convolutions用在一个56的深层GCN，应用在point clond semantic segmentation
  
---

- **Video Recognition/Prediction** 
    - 19-ICCV-oral-SlowFast Networks for Video Recognition, [pdf](https://arxiv.org/pdf/1812.03982.pdf), (code will be availabe)
    - 19-CVPR-Time-Conditioned Action Anticipation in One Shot, [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Time-Conditioned_Action_Anticipation_in_One_Shot_CVPR_2019_paper.pdf)
    - 19-CVPR-Peeking into the Future: Predicting Future Person Activities and Locations in Videos, [pdf](https://github.com/google/next-prediction), [tensorflow code](https://github.com/google/next-prediction)
        - 一个利用videos做行人的action prediction和trajectory prediction的统一模型。
        - 通过object detection，person key-point detection, scene segmentation, bounding boxes of objects and persons 的预训练模型（除了最后一个）来分别提appearance，motion，person-scene interaction, person-object interaction的visual feature。
    - 18-ECCV-Action Anticipation By Predicting Future Dynamic Images, [pdf](https://arxiv.org/abs/1808.00141)
        - 用dynamic images的重建L2 loss, dynamic images的分类loss， RGB frame的重建L2 loss训练
    - 18-ACCV-VIENA2: A Driving Anticipation Dataset,
[pdf](https://arxiv.org/abs/1810.09044), [dataset](https://sites.google.com/view/viena2-project/home)
        - 主要是提供了一个仿真的汽车意图有关的数据集
    - 17-ICCV-Encouraging LSTMs to Anticipate Actions Very Early, [pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Aliakbarian_Encouraging_LSTMs_to_ICCV_2017_paper.pdf), [theano+keras code](https://github.com/mangalutsav/Multi-Stage-LSTM-for-Action-Anticipation)
        - 使用了CNN提取visual feature (context), 使用Class Activation Map提取motion feature
