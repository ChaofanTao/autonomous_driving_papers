# Autonomous Driving Papers with Code
Papers and code collection about autonomous driving

- **A Famous Repo for Autonomous Driving**  
   [Awesome Autonomous Driving](https://github.com/autonomousdrivingkr/Awesome-Autonomous-Driving)
  
- **Trajectory Prediction**
    - 19-ICCV-The Trajectron: Probabilistic Multi-Agent Trajectory Modeling with Dynamic Spatiotemporal Graphs, [[pdf]](https://arxiv.org/pdf/1810.05993.pdf)
    - 19-ICCV-Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction, [[pdf]](https://arxiv.org/abs/1907.10178)
    - 19-ICCV-PRECOG: PREdiction Conditioned On Goals in Visual Multi-Agent Settings, [[pdf]](https://arxiv.org/pdf/1905.01296.pdf), [[project]](https://sites.google.com/view/precog)
    - 19-AAAI-oral-TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents, [[pdf]](https://arxiv.org/pdf/1811.02146.pdf), [[pytorch code]](https://github.com/huang-xx/TrafficPredict)
        - 提出 instance layer来建模实例之间的运动和相互影响； 用category layer来建模同类别实例之间的相似性。
        - 用统一的模型预测汽车、人、自行车的轨迹。  
    - 19-CVPR-precognition workshop Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, [[pdf]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Amirian_Social_Ways_Learning_Multi-Modal_Distributions_of_Pedestrian_Trajectories_With_GANs_CVPRW_2019_paper.pdf), [[pytorch code]](https://github.com/amiryanj/socialways)_
    - 18-CVPR-Social GAN Socially Acceptable Trajectories with GANs, [[pdf]](https://arxiv.org/pdf/1803.10892.pdf), [[pytorch code]](https://github.com/agrimgupta92/sgan)
        - 把trajectory prediction看成点列生成问题，所以用GAN-based architecture。
        - 在top k confident predictions用L2 loss, 增加variety。
    - 18-CVPR Trajnet Workshop -Convolutional Social Pooling for Vehicle Trajectory Prediction, [[pdf]](https://arxiv.org/abs/1805.06771)
        - Conv层+pooling层来代替social lstm里面的social pooling。方法是把lstm输出的hidden feature放在一个tensor里，tensor的列数是lane的个数，行数是输入总高度/单个car的长度，深度数是hidden dimension。
    - 16-CVPR-Social LSTM Human Trajectory Prediction in Crowded Spaces, [[pdf]](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf), [[pytorch code]](https://github.com/quancore/social-lstm)
        - 提出social pooling对空间上相近的人的关系建模，方法是在人工设置大小的grid里，把同一个grid的hidden feature累加。


- **3D Tracking/Object Detection/Segmentation/Depth Estimation**
    - 19-ICCV-Deep HoughVoting for 3D Object Detection in Point Clouds, [[pdf]](https://arxiv.org/abs/1904.09664)
    - 19-ICCV-Joint Monocular 3D Vehicle Detection and Tracking, [[pdf]](https://arxiv.org/abs/1811.10742), [[pytorch code]](https://github.com/ucbdrive/3d-vehicle-tracking)
    - 19-ICCV-oral-Deep Hough Voting for 3D Object Detection in Point Clouds, [[pdf]](https://arxiv.org/abs/1904.09664)
    - 19-ICCV-PU-GAN: a Point Cloud Upsampling Adversarial Network, [[pdf]](https://arxiv.org/pdf/1907.10844.pdf)
    - 19-CVPR-ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving, [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Song_ApolloCar3D_A_Large_3D_Car_Instance_Understanding_Benchmark_for_Autonomous_CVPR_2019_paper.pdf)
    - 19-CVPR-Stereo R-CNN based 3D Object Detection for Autonomous Driving, [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.pdf), [[pytorch code]](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)
    - 19-arXiv-A Baseline for 3D Multi-Object Tracking, [[pdf]](https://arxiv.org/pdf/1907.03961v2.pdf), [[code]](https://github.com/xinshuoweng/AB3DMOT)
         - SOTA for 3D Multi-Object Tracking on KITTI
    - 19-CVPR-PointPillars: Fast Encoders for Object Detection from Point Clouds, [[PDF]](https://arxiv.org/pdf/1812.05784v2.pdf), [[pytorch code]](https://github.com/nutonomy/second.pytorch)
    - 19-CVPR-LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving, [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.pdf)
    - 19-CVPR-Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving, [[pdf]](https://arxiv.org/pdf/1812.07179v5.pdf), [[pytorch code]](https://github.com/mileyan/pseudo_lidar)
    - 19-ICCV-3D-RelNet: Joint Object and Relational Network for 3D Prediction, [[pdf]](https://arxiv.org/pdf/1906.02729.pdf), [[pytorch code]](https://nileshkulkarni.github.io/relative3d/)
    - 19-ICCV-3D Point Cloud Learning for Large-scale Environment Analysis and Place Recognition, [[pdf]](https://arxiv.org/pdf/1812.07050.pdf)
    - 19-ICCV-oral-Can GCNs Go as Deep as CNNs? [[pdf]](https://arxiv.org/pdf/1904.03751.pdf), [[tensorflow code]](https://github.com/lightaime/deep_gcns), [[pytorch code]](https://github.com/lightaime/deep_gcns_torch)
        - 把CNN的residual/dense connection和dilated convolutions用在一个56的深层GCN，应用在point clond semantic segmentation
    - 18-arXivComplex-YOLO: Real-time 3D Object Detection on Point Clouds, [[pdf]](https://arxiv.org/pdf/1803.06199v2.pdf), [[pytorch code]](https://github.com/AI-liu/Complex-YOLO)
    - 17-CVPR-Multi-View 3D Object Detection Network for Autonomous Driving, [[pdf]](https://arxiv.org/pdf/1611.07759v3.pdf), [[tensorflow code]](https://github.com/bostondiditeam/MV3D)
    - 17-arXiv-SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud, [[pdf]](https://arxiv.org/pdf/1712.02294v4.pdf), [[tensorflow code]](https://github.com/kujason/avod)
  

- **Action Recognition/Prediction** 
    - __A famous repo for action recognition__: [[Awesome Action Recognition]](https://github.com/jinwchoi/awesome-action-recognition)
    - 19-ICCV-oral-SlowFast Networks for Video Recognition, [[pdf]](https://arxiv.org/pdf/1812.03982.pdf), (code will be available)
    - 19-CVPR-Time-Conditioned Action Anticipation in One Shot, [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Time-Conditioned_Action_Anticipation_in_One_Shot_CVPR_2019_paper.pdf)
    - 19-CVPR-Peeking into the Future: Predicting Future Person Activities and Locations in Videos, [[pdf]](https://github.com/google/next-prediction), [[tensorflow code]](https://github.com/google/next-prediction)
        - 一个利用videos做行人的action prediction和trajectory prediction的统一模型。
        - 通过object detection，person key-point detection, scene segmentation, bounding boxes of objects and persons 的预训练模型（除了最后一个）来分别提appearance，motion，person-scene interaction, person-object interaction的visual feature。
   - 19-ICCV-Exploring the Limitations of Behavior Cloning for Autonomous Driving [[pdf]](https://arxiv.org/pdf/1904.08980.pdf), [[python code]](https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md)
    - 18-ECCV-Action Anticipation By Predicting Future Dynamic Images, [[pdf]](https://arxiv.org/abs/1808.00141)
        - 用dynamic images的重建L2 loss, dynamic images的分类loss， RGB frame的重建L2 loss训练
    - 19-arXiv-Temporal Recurrent Networks for Online Action Detection, [[pdf]](https://arxiv.org/pdf/1811.07391.pdf), [[pytorch code]](https://github.com/rajskar/CS763Project)
         - It jointly models the historical and future temporal context under the constraint of the online setting
    - 17-ICCV-Online Real-time Multiple Spatiotemporal Action Localisation and Prediction, [[pdf]](https://arxiv.org/pdf/1611.08563v6.pdf), [[pytorch code]](https://github.com/gurkirt/realtime-action-detection)
         - real-time SSD (Single Shot MultiBox Detector) CNNs to regress and classify detection boxes in each video frame potentially containing an action of interest
         - propose an online algorithm to incrementally construct and label "action tubes" from the SSD frame level detections
    - 17-ICCV-Temporal Action Detection with Structured Segment Networks, [[pdf]](https://arxiv.org/abs/1704.06228), [[pytorch code]](https://github.com/yjxiong/action-detection)
    - 17-ICCV-Encouraging LSTMs to Anticipate Actions Very Early, [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Aliakbarian_Encouraging_LSTMs_to_ICCV_2017_paper.pdf), [[theano+keras code]](https://github.com/mangalutsav/Multi-Stage-LSTM-for-Action-Anticipation)
        - 使用了CNN提取visual feature (context), 使用Class Activation Map提取motion feature

- **Video Prediction/Generation**  
    - 18-ICML-Hierarchical Long-term Video Prediction without Supervision, [[pdf]](http://web.eecs.umich.edu/~honglak/icml2018-unsupHierarchicalVideoPred.pdf), [[tensorflow code]](https://github.com/brain-research/long-term-video-prediction-without-supervision)  
    - 18-arXiv-Stochastic Adversarial Video Prediction, [[pdf]](https://arxiv.org/abs/1804.01523), [[tensorflow code]](https://github.com/alexlee-gk/video_prediction)  
    - 18-arXiv-Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning, [[pdf]](https://arxiv.org/abs/1605.08104), [[keras code]](https://github.com/coxlab/prednet)  
    - 18-NeurlPS-Learning to Decompose and Disentangle Representations for Video Prediction, [[pdf]](https://arxiv.org/abs/1806.04166), [[pytorch+pyro code]](https://github.com/jthsieh/DDPAE-video-prediction)  
    - 16-arXiv-Learning a Driving Simulator, [[pdf]](http://arxiv.org/abs/1608.01230), [[tensorflow code]](https://github.com/commaai/research)  
    - 16-arXiv-Deep Multi-Scale Video Prediction Beyond Mean Square Error, [[pdf]](https://arxiv.org/abs/1511.05440), [[tensorflow code]](https://github.com/dyelax/Adversarial_Video_Generation)  

- **Dataset**
    - [[HDD]](https://usa.honda-ri.com/hdd) 18-CVPR. [[pdf]](https://arxiv.org/pdf/1811.02307.pdf). A dataset for (own) driving scene understanding. Nearly 104 hours of 137 driving sessions in the San Francisco Bay Area. The dataset was collected from a vehicle with a front-facing camera, and includes frame-level annotations of 11 goal-oriented actions (e.g., intersection passing, left turn, right turn, etc.) The dataset __also__ includes readings from a variety of non-visual sensors collected by the instrumented vehicle’s Controller Area Network (CAN bus).
    - [[KITTI]](http://www.cvlibs.net/datasets/kitti/index.php) Tasks of interest are: stereo evaluation, optical flow evaluation, depth estimation, visual odometry, 3D object detection and 3D tracking, semantic segmentation
    - [[APOLLO Scape]](http://apolloscape.auto/) Scene Parsing ,Car Instance,Lane Segmentation,Self Localization,Trajectory, Detection/Tracking, Stereo
    - [[nuScenes]](https://www.nuscenes.org) The first large-scale dataset to provide data from the entire sensor suite of an autonomous vehicle (6 cameras, 1 LIDAR, 5 RADAR, GPS, IMU). The goal of nuScenes is to look at the entire sensor suite. The full dataset includes approximately 1.4M camera images, 390k LIDAR sweeps, 1.4M RADAR sweeps and 1.4M object bounding boxes in 40k keyframes.
    - [[Caltech Lanes]](http://www.mohamedaly.info/datasets/caltech-lanes) The archive below inlucdes 1225 individual frames as taken from a camera mounted on Alice in addition to the labeled lanes. The dataset is divided into four individual clips: cordova1 with 250 frames, cordova2 with 406 frames, washington1 with 337 frames, and washington2 with 232 frames. 
    - [[Virtual KITTI]](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/) 2D/3D object detection, multi-object tracking
    - [[Berkeley DeepDrive]](https://bdd-data.berkeley.edu/) Object detection, instance segmentation ,drivable decision, lane marking. Explore 100,000 HD video sequences of over 1,100-hour driving experience across many different times in the day, weather conditions, and driving scenarios. Our video sequences also include GPS locations, IMU data, and timestamps.
    - [[VIENA2]](https://sites.google.com/view/viena2-project/home) Synthetic driving data for driving manoeuvre, accidents, pedestrian intentions and front car intentions. 15K HD videos with frame size of 1920x1280, corresponding to 2.25M annotated frames. Each video contains 150 frames captured at 30fps depicting a single action from one scenario.

- **Challenge**
    - [[19-CVPR WAD]](http://wad.ai/challenge.html) Workshop on AD

- **Miscellaneous**
   - Paper with Code for [Autonomous Driving](https://paperswithcode.com/task/autonomous-driving/codeless) and [Self-Driving Cars
](https://paperswithcode.com/task/self-driving-cars)
   - [[MMAcition]](https://github.com/open-mmlab/mmaction) 
