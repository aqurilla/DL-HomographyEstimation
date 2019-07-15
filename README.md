P1 - My AutoPano README
==================

Phase 1
=======

Wrapper.py can be run by calling python3 Wrapper.py with the following arguments:
	--NumFeatures [Number of corners to consider per component image. Note that a panorama-in-progress containing N component images will have N x NumFeatures corners]
	--ImDir [A unix expression (interpretable by glob) for the images to attempt to stitch. Images are sorted in alphanumeric order by name and are stitched in that order]
	--RansacIters [Number of iterations of RANSAC to perform]

Note that images are displayed and saved using calls to displayAndSave on lines 66,80,90,101,116, and 135. By default, images are saved to a directory '../Output', as specified on lines 197 and 192 of helpers.py. Filenames should be changed on the aforementioned lines of Wrapper.py


Phase 2
=======
There are two training scripts - one for the supervised (Train.py) and one for the unsupervised (Train_unsup.py). 

For running the code, the /Data/ and /Checkpoints/ folders are placed like such in the filetree-

YourDirectoryID_p1.zip
|   Phase2
|   ├── Data
|   |   ├── Train
|   |   ├── Test
|   |   ├── Val
|   |   └── Pano
|   |       ├── TestSet1
|   |       ├── TestSet2
|   |       ├── TestSet3
|   |       └── TestSet4
|   ├── Checkpoints
|   ├── Code
|   |   ├── Train.py
|   |   ├── Train_unsup.py
|   |   ├── Test.py
|   |   ├── Wrapper.py
|   |   └── Misc
|   |   └── Network
├── Report.pdf
└── README.md


Supervised DL training
======================
For supervised training, first run data_gen.py with the arguments stating which dataset to generate images from. The images and their H4Pt labels are saved in the folder, which is then used by Train.py for training. 'MaxImVal' should be set to 5000 for the training image set, and 1000 for test and validation sets.

$ python2 data_gen.py --DataSet=Train --MaxImVal=5000
$ python2 data_gen.py --DataSet=Test --MaxImVal=1000
$ python2 data_gen.py --DataSet=Val --MaxImVal=1000

After generating Training, Validation and Testing datasets in this manner, training can be done. Arguments are those already present in the provided template training file

$ python2 Train.py --NumEpochs=50 --LogsPath=Logs/7/ --MiniBatchSize=64 --CheckPointPath=../Checkpoints/7/


Unsupervised DL training
========================
For the unsupervised training script (Train_unsup.py), data pregeneration is not required, as the 'online_data_gen.py' script generates data in a streaming manner for the program. The program expects a maximum of 5000 images in the /Train/ dataset, and 1000 in the /Test/ dataset 

$ python2 Train_unsup.py --NumEpochs=50 --LogsPath=Logs/8/ --MiniBatchSize=64 --CheckPointPath=../Checkpoints/8/


Testing
=======
Testing uses the pregenerated 'im_Test.npy' and 'h4_Test.npy' image dataset and labels, which were created initially. Additionally, the model path that should be tested is provided as the ModelPath argument. Also ModelType argument takes whether it is the supervised or unsupervised model

$ python2 Test.py --ModelPath='../Checkpoints/11/49model.ckpt'


Panorama generation
===================
For panorama generation, Wrapper.py is used. It takes as input ModelPath, which is the model used for panorama generation, and ImDir, which is the image directory containing the panorama to be tested

$ python2 Wrapper.py --ModelPath='../Checkpoints/11/49model.ckpt' --ImDir='../Data/Pano/TestSet1/*.jpg'


