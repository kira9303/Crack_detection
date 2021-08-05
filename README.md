# Crack_detection
Crack detection and classification using machine learning

The basic idea of the project is to come up with a robust machine learning model that can help UAV's in detecting the cracks in high altitude buildings and further classifying the detected cracks into "major" & "minor" cracks. I've used two models here, each capable of binary classification. The first model predicts whether the given image has a crack present in it or no and, the second image further classifies the cracked image into two categories, i.e "major_crack" or "minor_crack". The execution process is explained below in detail. The the dataset can be found on kaggle and the link to it is mentioned below. The first model runs on the dataset and the second model runs on custom selected images + some web scraped images of major and minor cracks from google. So, to run the second training script runs on custom selected images from the first dataset and the link to this dataset can be found below. Download the dataset and change the directory path in your second crack training script to point towards this dataset.




2) Process of execution

1. I've used four (4) scripts in total to execute the entire project. Install all the scripts in a directory.
2. download the dataset folders from the drive link mentioned below and paste them in the same directory as the scripts. 
3. Run the training scripts first. After the model is created at the end of the training scripts, it will be saved in the same directory where your scripts are.
4. After you have two .h5 files (models) in your folder, run the "Crack_implementation.py" script and enter the path to any image within your local machine. That's it!


Link to download the dataset:--
https://drive.google.com/drive/folders/1j8racJ1IlKnpfqNHcdXQPMU563deeIiN?usp=sharing



