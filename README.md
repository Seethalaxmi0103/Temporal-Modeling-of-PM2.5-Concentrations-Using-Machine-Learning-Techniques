<b>Project Overview</b>

Air pollution is a major environmental and public health concern worldwide. Fine particulate matter (PM2.5) is particularly harmful because it can penetrate deep into the lungs and bloodstream, leading to respiratory and cardiovascular diseases.
This project implements a machine learning-based predictive analytics system to estimate PM2.5 air pollution levels using historical air quality data. The model uses temporal features such as year, month, day, hour, and day-of-week to capture patterns in pollution levels.
A Random Forest Regression model is trained to analyze the dataset and predict PM2.5 concentrations. The system also generates 24-hour future predictions, which can support environmental monitoring and smart city air quality management.

<b>Dataset:</b><br> 
The project uses the dataset: air-quality-india.csv<br>
The dataset contains historical air quality observations including time-related attributes and PM2.5 concentration levels.(Downloaded from Kaggle)

Important Columns

| Column    | Description                  |
| --------- | ---------------------------- |
| year      | Year of observation          |
| month     | Month of observation         |
| day       | Day of observation           |
| hour      | Hour of observation          |
| timestamp | Date and time of measurement |
| pm2_5     | PM2.5 concentration (µg/m³)  |


<b>Project Workflow</b>

1. Import Libraries:
Essential libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn are imported for data processing and modeling.
2. Load Dataset:
The dataset is loaded using Pandas:
df = pd.read_csv("air-quality-india.csv")
3. Data Cleaning:
Column names are standardized
Missing PM2.5 values are replaced using the mean value
4. Timestamp Processing:
Timestamp data is converted into datetime format for time-based analysis.
5. Feature Engineering:
New features are extracted:
day_of_week
is_weekend
These help the model learn weekly pollution patterns.
6. Train-Test Split:
The dataset is divided into:
80% training data
20% testing data
7. Model Training:
A Random Forest Regressor with 200 trees is used for training.
model = RandomForestRegressor(n_estimators=200)
8. Model Evaluation:
Performance is evaluated using:
Mean Squared Error (MSE)
R² Score
9. Visualization:
The project plots:
Actual vs predicted PM2.5 values
Future PM2.5 prediction graph
10. Model Saving:
The trained model is saved as:
air_quality_model.pkl
This allows reuse without retraining.
11. Future Prediction:
The system generates PM2.5 predictions for the next 24 hours based on temporal inputs.

<b>Output:</b><br>
The program produces:
Model evaluation metrics,
Actual vs predicted PM2.5 graph,
24-hour PM2.5 forecast plot,
Saved trained model file

<b>Example outputs:</b><br>
Mean Squared Error: 20.15
R² Score: 0.967


<b>Installation</b><br>

Step 1: Clone Repository
git clone https://github.com/yourusername/air-quality-prediction.git<br>
Step 2: Navigate to Project Folder
cd air-quality-prediction<br>
Step 3: Install Dependencies
pip install -r requirements.txt

<b>Running the Project</b><br>

Run the Python script:
python air_quality_prediction.py

Make sure the dataset file air-quality-india.csv is placed in the same directory as the script.

Technologies Used:
Python,
Pandas,
NumPy,
Matplotlib,
Scikit-learn,
Joblib
