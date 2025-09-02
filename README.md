# deep_learning_skin_cancer_detection_multi_type_ISIC_2018....

A deep-learning pipeline for skin lesion classification using ISIC dataset with multiple deep learning cnn algorithms and advanced preprocessing including multithreaded loading, augmentation, and performance evaluation.

## Environment Setup

Install Python 3.8+ and required packages 

            or

pip install -r requirements.txt

###load and resize
python data/loader.py

# To preprocess, Normalize, Encode, and Split Data, Balance and augment data
python data/pipeline.py
python data/augmentation.py



#for training, choose any model!..
python training/train.py --model <model_name>

#for inference on test data
python inference.py --model_path saved_models/densenet201.h5

##Make sure the following are in place before running:
1. saved_models/ directory exists with trained .h5 files.
2. Your environment has TensorFlow, Streamlit, Seaborn, PIL, and Matplotlib installed.


#To run single predict image file
python predict_single.py --image_path path/to/image.jpg --model_path saved_models/densenet201.h5



## to run the inferece using the front/back end, uisng the streamlit lib,
streamlit run ./app/gui.py

What and How This App Works:
1. Loads trained model dynamically from the saved_models/ directory.
2. Processes an uploaded dermoscopic image using TensorFlow preprocessing.
3. Runs inference on the selected model.
4. Displays the predicted skin cancer class and confidence.
5. Visualizes class confidence using a bar plot via Seaborn.
6. Fully integrated with the earlier single_predict logic for robustness. (In Progress)


## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
