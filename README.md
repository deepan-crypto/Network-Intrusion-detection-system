 Network Anomaly Detection using a Random Forest Classifier
This project builds and evaluates a machine learning model for network anomaly detection using the NSL-KDD dataset. The goal is to classify network traffic into five categories: Normal, DoS, Probe, Privilege, and Access attacks. A Random Forest Classifier is trained to perform this multi-class classification, achieving a final accuracy of over 99%.

📊 Dataset
Dataset: The project uses the NSL-KDD dataset, a benchmark dataset for intrusion detection research. It was downloaded from a Hack The Box Academy URL.

Features: Each record in the dataset represents a network connection and contains 41 features describing the traffic, such as duration, protocol type, service, and various traffic statistics.

Labels: The connections are labeled as either 'normal' or a specific type of attack.

⚙️ Project Workflow
The notebook follows a structured approach to build, train, and evaluate the classifier.

1. Data Loading and Preprocessing
The NSL-KDD dataset is downloaded and loaded into a pandas DataFrame with predefined column names.

Categorical features like protocol_type and service are converted into a numerical format using one-hot encoding with pandas.get_dummies.

The final feature set is constructed by combining these encoded features with the original numeric features of the dataset.

2. Feature Engineering
The original, highly specific attack labels are mapped into broader categories for a multi-class classification task. A new target column, attack_map, is created with the following mapping:

0: Normal

1: DoS (Denial of Service) attacks

2: Probe (Probing/Scanning) attacks

3: Privilege (Privilege Escalation) attacks

4: Access (Unauthorized Access) attacks

3. Model Training and Evaluation
Data Splitting: The dataset is split into three parts: a training set, a validation set, and a test set. This ensures the model is evaluated on data it has never seen before.

Model Selection: A Random Forest Classifier from the scikit-learn library is used for this task.

Training: The model is trained on the training data using the .fit() method.

Evaluation: The model's performance is first checked on the validation set and then on the final test set using multiple metrics, including accuracy, precision, recall, and F1-score. A confusion matrix and a detailed classification report are generated to analyze its performance across different attack categories.

📈 Results
The trained Random Forest model demonstrates excellent performance in identifying network anomalies.

Final Test Set Accuracy: 99.49%

Final Test Set F1-Score (Weighted Avg): 0.9947

Test Set Confusion Matrix
The confusion matrix shows a high number of correct predictions along the main diagonal. The model is exceptionally good at identifying Normal traffic, DoS attacks, and Probe attacks.

Performance Analysis
As shown in the classification report, while the overall accuracy is very high, the model's performance on the Privilege class is notably lower (recall of 0.24). This is a common result with this dataset and is due to the very small number of training samples for this attack category, making it difficult for the model to learn its patterns effectively.

