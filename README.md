Health Care ChatBot - README

This Python code implements a health care chatbot that predicts possible medical conditions based on symptoms provided by the user. The chatbot utilizes a Decision Tree Classifier and Support Vector Machine (SVM) for prediction.

Requirements:
- Python 3.x
- Required Python libraries: re, pandas, pyttsx3, sklearn, numpy, csv

Usage:
1. Ensure that the required Python libraries are installed.
2. Place the training and testing data in the specified file paths ("Data/Training.csv" and "Data/Testing.csv").
3. Update the file paths in the code if necessary.
4. Run the code.

Functionality:
- The chatbot prompts the user for their name and initiates a conversation.
- The user is asked to input their symptoms.
- The chatbot predicts possible medical conditions using a trained Decision Tree Classifier.
- If the confidence of the prediction is below a certain threshold, the chatbot asks for additional symptoms.
- The severity of symptoms is calculated, and recommendations are provided based on the severity.
- The chatbot outputs the predicted medical conditions, descriptions, and precautions to the user.

Note:
- The code relies on CSV files for symptom descriptions, severity levels, and precautions. Make sure the required files ("MasterData/symptom_Description.csv", "MasterData/symptom_severity.csv", "MasterData/symptom_precaution.csv") are present and properly formatted.

Author: [Sneha Nagarch]


