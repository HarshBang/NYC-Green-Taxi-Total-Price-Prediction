# NYC Green Taxi Fare Predictor

This project demonstrates an end-to-end machine learning solution to predict the total fare for a New York City Green Taxi trip. Using various trip attributes, a Random Forest Regressor model was trained and deployed as an interactive web application using Streamlit, allowing for real-time fare estimations.

## 🚀 Live Demo [(link)](https://nyc-green-taxi-total-price-prediction.streamlit.app/)
<img width="1027" height="767" alt="Image" src="https://github.com/user-attachments/assets/b9a1bc5d-dd91-4f38-b5b4-527fc77166ab" />


## 🛠️ Technologies Used

The project was built using a combination of tools and libraries for data processing, model training, and deployment:

* **Core Language & Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Model Persistence:** Pickle, Joblib
* **Modeling:** Random Forest Regressor, Feature Importance Analysis

## 📋 Project Workflow

The project was executed in a structured workflow, from data exploration to final deployment:

1.  **Data Cleaning**: The initial dataset was cleaned by handling null values and removing outliers, such as trips with zero distance or negative fare amounts. 
2.  **Feature Engineering**: New, valuable features were created from the existing data. This included extracting the `hour` and `weekday` from timestamps and calculating `trip_duration` from pickup and drop-off times.
3.  **Data Preprocessing**:
    * **Multicollinearity Handling**: The `fare_amount` feature was dropped to prevent data leakage, as it was highly correlated with the target variable, `total_amount`. 
    * **Scaling**: All numerical input features were scaled using `MinMaxScaler` to normalize their range. The target variable was not scaled. 
4.  **Model Training & Feature Selection**:
    * An initial model was trained using all 18 features. 
    * `feature_importances_` from the Random Forest model was used to identify and select the **top 10 most influential features**. 
    * The final model was retrained exclusively on these top 10 features for improved efficiency and generalization. 
5.  **Deployment**: The final, optimized model was deployed as an interactive web application using Streamlit. 

## 📊 Dataset

The dataset was sourced from the **NYC Green Taxi Trip Records** for January 2024.

* **Target Variable**: `total_amount` - The complete fare for a single taxi ride. 
* **Initial Features**: The dataset began with 18 features, including `trip_distance`, `tip_amount`, pickup/drop-off zones, and timestamps. 
* **Final Features**: The deployed model uses the top 10 most important features for prediction: `trip_distance`, `tip_amount`, `PUZone`, `DOZone`, `RatecodeID`, `passenger_count`, `trip_duration`, `hour`, `weekday`, and `mta_tax`. 

## 🤖 Model Performance

Several regression models were trained and evaluated. The Random Forest Regressor demonstrated the best performance and was selected for the final application.
| Model                 | Test Accuracy |
| :-------------------- | :------------ |
| **Random Forest** | **90.79%** |
| Gradient Boosting     | 89.24%        |
| Decision Tree         | 80.86%        |
| K-Nearest Neighbors   | 42.17%        |
| Logistic Regression   | 41.75%        |

The final model achieved an **R² Score of approximately 0.89** on the test data, indicating a strong fit.

## 🖥️ Web Application Features

The Streamlit web app provides an intuitive interface for fare prediction:

* **User Input**: Users can input values for the top 10 trip features using interactive sliders and input boxes.
* **Real-Time Prediction**: The inputs are preprocessed using the saved scaler and fed to the trained model (`best_model_rf_top10.pkl`) to generate a prediction. 
* **Dynamic Display**: The estimated total fare is displayed instantly on the screen.

## 📁 File Structure
NYC-Taxi-Fare-Predictor/

├── app.py                     (Streamlit application code)

├── model_building.ipynb       (Jupyter Notebook for EDA, preprocessing, and model training)

├── best_model_rf_top10.pkl    (Saved Random Forest model (trained on top 10 features))

├── important_features_top10.pkl (Saved list of the selected top 10 features)

├── num_scaler.pkl             (Saved scaler object for numerical features)

└── requirements.txt           (Required libraries for installation)

## 📈 Future Improvements

* Incorporate more granular time-based features from pickup and drop-off datetimes. 
* Experiment with more advanced models like XGBoost or LightGBM. 
* Use geolocation clustering to reduce the dimensionality of pickup and drop-off zones.
* Expand the application to support Yellow Taxi or For-Hire Vehicle (FHV) datasets. 
