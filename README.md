# deep_learning_skin_cancer_detection_multi_type_isic2018
A deep learning pipeline for skin lesion classification using ISIC dataset with multiple deep learning cnn algorithms and advanced preprocessing including multithreaded loading, augmentation, and performance evaluation.

###load and resize
python data/loader.py

# To preprocess, Normalize, Encode, and Split Data, Balance and augment data
python data/pipeline.py
python data/augmentation.py



#for training, choose any model!..
python training/train.py --model <model_name>

#for inference on test data
python inference.py --model_path saved_models/densenet201.h5

#To run ssingle predict image file
python predict_single.py --image_path path/to/image.jpg --model_path saved_models/densenet201.h5

