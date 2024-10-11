# BoT-FaceSORT
**BoT-FaceSORT: Bag-of-Tricks for Robust Multi-Face Tracking in Unconstrained Videos**

<!-- ## Structure
```
botfacesort
  |——detector
    |——scrfd
  |——inference
    |——ChokePoint
      |——scripts
    |——MovieShot
      |——scripts
    |——Music
      |——scripts
    inference_all.sh
  |——test_data
    |——MovieShot
      |——MovieShot-all
      |——seqmaps
    |——ChokePoint
      |——ChokePoint-all
      |——seqmaps
    |——Music
      |——seqmaps
  |——tracker
    |——appearance
    |——configs
    |——motion
    |——postprocessing
    |——trackers
    |——utils
    __init__.py
    tracker_zoo.py
  |——trackeval
  |——utils
  eval.py
  requirements.txt
  track.py
``` -->
## Abstract
Multi-face tracking (MFT) is a subtask of multi-object tracking (MOT) that focuses on detecting and tracking multiple faces across video frames. Modern MOT trackers adopt the Kalman filter (KF), a linear model that estimates current motions based on previous observations. However, these KF-based trackers struggle to predict motions in unconstrained videos with frequent shot changes, occlusions, and appearance variations. To address these limitations, we propose **BoT-FaceSORT**, a novel MFT framework that integrates a shot change detection, a shared feature memory, and an adaptive cascade matching strategy for robust tracking. It detects shot changes by comparing the color histograms of adjacent frames and resets KF states to handle discontinuities. Additionally, we introduce MovieShot, a new benchmark of challenging movie clips to evaluate MFT performance in unconstrained scenarios. We also demonstrate the superior performance of our method compared to existing methods on three benchmarks, while an ablation study validates the effectiveness of each component in handling unconstrained videos.

## Data Preparation
We use [MOT15](http://arxiv.org/abs/1504.01942) format for each dataset as default.

Please make the root dataset path as ``test_data``.
```shell
mkdir test_data
```

### 1. MovieShot Dataset [[Google Drive]](https://drive.google.com/file/d/1AIy81BU6Su4AohDJUtllhud1cBtypbp9/view?usp=sharing)
- We provide each source of YouTube video ID of MovieShot at utils/constants.py **SEQUENCES** variable. Please download each video in 720p resolution to test_data/MovieShot/videos.
- Please download the dataset from google drive link on the root of the repo and extract it under test_data/MovieShot.

```shell
unzip MovieShot.zip -d test_data/MovieShot && rm MovieShot.zip
```
  
#### Generate seqinfo for each sequence as follows:
```shell
python utils/build_movieshot.py
```

#### Final dataset structure 
```
|——test_data
  |——MovieShot
    |——MovieShot-all
    |——seqmaps
    |——shot_changes
    |——videos
  ```
### 2. Music Dataset [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_26)|[[Project Page]](https://sites.google.com/site/shunzhang876/eccv16-face-tracking)
- Download videos and ground truths from the project page.
- Extract videos and ground truths to test/data/Music. (Video: videos/, ground truths: gt/)
- **Important: Please rename stickwithu_gt.xml to PussycatDolls_gt.xml and Tara_gt.xml to T-ara_gt.xml.**"
```
|——test_data
  |——Music
    |——gt
      |——Apink_gt.xml
      |——...    
    |——...  
    |——videos
      |——Apink.mp4
      |——...
```
#### Convert XML format to MOT15 format and generate seqinfo as follows:

```shell
python utils/build_music.py \
       --gt-folder test_data/Music/gt \
       --video-folder test_data/Music/videos \
       --data-folder test_data/Music/Music-all
```

#### Final dataset structure 
```
|——test_data
  |——Music
    |——gt
      |——Apink_gt.xml
      |——...    
    |——Music-all
      |——Apink
        |——gt
          |——gt.txt
        seqinfo.ini
      |——...
    |——videos
      |——Apink.mp4
      |——...
```
### 3. ChokePoint Dataset [[Paper]](https://ieeexplore.ieee.org/document/5981881)|[[Project Page]](https://arma.sourceforge.net/chokepoint/)
- Please download two sequences(P2E_S5, P2L_S5), which are recorded with a crowded scenario.

#### Download and extract
Please download each sequence to the root of the repo.
```shell
mkdir ChokePoint
find . -name '*.tar.xz' -exec tar -xvf {} -C ChokePoint \;
```

```shell
find ChokePoint -name '*.tar.xz' -exec tar -xvf {} -C ChokePoint \;
```

#### Merge images into video
We currently unsupport inference on image sequences. Therefore, please merge images into videos as follows:
```shell
sh utils/build_chokepoints.sh
```
#### Remove temporal files
```shell
rm -rf ChokePoint && rm *.tar.xz
```

#### Final dataset structure 
```
|——test_data
  |——ChokePoint
    |——ChokePoint-all
    |——seqmaps
    |——videos
```


## Environments
[![Python 3.8.5](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-385/)
- Ubuntu 20.04.6
- AMD Ryzen 9 7950X @ 4.5Ghz
- NVIDIA GeForce RTX 4090
- Samsung DDR5 5600MHz 32GB * 4EA
- CUDA 11.8
- Python 3.8.5
- PyTorch 2.0.1


## Install Dependencies
Please install Numpy and Setuptools first, and install the remaining requirements.

```shell
pip install numpy==1.24.1 setuptools==72.1.0
pip install -r requirements.txt
```

## Tracking
For easier tracking, we provide the shell scripts for each experiment.

Scripts are placed in **inference/{Database}/scripts/{tracking-method}.sh**

For `Database`, we support **MovieShot**, **Music**, and **ChokePoint**.

For tracking method, we support **botfacesort**, **botfacesort_sc**, **botfacesort_sm**, **botsort**, **bytetrack**, **deepocsort**, **deepsort**, **hybridsort**, **ocsort**, and **strongsort**.

If you want to perform tracking for all trackers in specific Database, Please run inference/{Database}/scripts/tracker_all.sh


- **Run all trackers on the MovieShot dataset**
  ```shell
  sh inference/MovieShot/scripts/tracker_all.sh
  ```

- **Run BoT-FaceSORT on the Music dataset**
  ```shell
  sh inference/Music/scripts/botsort.sh
  ```

We also provide scripts for ablation study in MovieShot and Music dataset.

The scripts are placed in inference/{Database}/scripts/ablation/botfacesort_{method}.sh.

For method, we support **base**, **sc** (shot change detection module), and **sm** (shared feature memory module).

- **Run base ablation study on MovieShot**
  ```shell
  sh inference/MovieShot/scripts/ablation/botfacesort_base.sh
  ```

- **Run ablation study for shot change detection module on Music**
  ```shell
  sh inference/Music/scripts/ablation/botfacesort_sc.sh
  ```

If you want to run tracking + ablation study, Please run inference/{Database}/scripts/run_all.sh
- **Run all tracking + ablation study on MovieShot**
  ```shell
  sh inference/MovieShot/scripts/run_all.sh
  ```

For convenience, we also provide the script to run all experiments within the paper at once.
- **Run all experiments on whole dataset**
  ```shell
  sh inference/inference_all.sh
  ```

## Evaluation
For evaluate HOTA, MOTA, IDF1 Score, we use the official code [TrackEval](https://github.com/JonathonLuiten/TrackEval) to evaluate the results on the MovieShot, Music, and ChokePoint dataset.

#### HOTA, MOTA, IDSW, IDF1 Evaluation
```shell
python eval.py \
--DO_PREPROC False \
--BENCHMARK {Database} \
--TRACKERS_TO_EVAL {tracking-method} \
--TRACKERS_FOLDER 'inference/{Database}/results' \
--TRACKER_SUB_FOLDER '' \
--GT_FOLDER 'test_data/{Database}'
```
#### FPS Evaluation
```shell
python utils/fps.py \
--database {Database} \
--tracker {tracking-method} \
--sample
```

#### Example: Evaluate BoT-FaceSORT on MovieShot dataset
  ```shell
  python eval.py \
  --DO_PREPROC False \
  --BENCHMARK MovieShot \
  --TRACKERS_TO_EVAL botfacesort \
  --TRACKERS_FOLDER 'inference/MovieShot/results' \
  --TRACKER_SUB_FOLDER '' \
  --GT_FOLDER 'test_data/MovieShot'
  ```
  ```shell
  python utils/fps.py \
  --database MovieShot \
  --tracker botfacesort
  ```

## Tracking on Your Video
```shell
python track.py -p {your_video_path} \
 --tracking-method botfacesort--sc \
 --sm \
 --conf 0.5 \
 --nms 0.7 \
 --device 0
```

If you want to show your tracking results, just add `--display` option into command.

## Citation
TBA

## Acknowledgement
A large part of the codes, ideas and results are motivated from [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), 
[BoxMOT](https://github.com/mikel-brostrom/boxmot), [InsightFace](https://github.com/deepinsight/insightface) and [AdaFace](https://github.com/mk-minchul/AdaFace). Thanks for their great work!
