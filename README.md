# Disease_Classification_DL
This is a practical Deep learning project for image classification.
•	API-based model is developed using Fastapi, CNN, data augmentation, resizing and rescaling on the tomato image dataset from Kaggle. \n
•	Running by tf serving on localhost with the output class & % confidence.
The requirements are there in the requirements.txt file.
use pip install requirement_name to install the specific requirement.
To run the model, open terminal and change directory to location on tomato_api.py and run uvicorn tomato_api:app --reload.
Either use localhost or Postman to upload and check the accuracy of the model. Relevant screenshots are available in the repository
