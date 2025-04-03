# **Delhi Electric Power Demand Peaks Prediction**  

Predicts daily peak electric power demand to help grid operators optimize resources, prevent overloads, and improve efficiency. Supports sustainability by reducing waste, enhancing renewable integration, and lowering carbon emissions. Contributes to a stable, smart, and resilient energy infrastructure



## **Overview**  
This project aims to develop a machine learning model to predict **daily peak power demand** for **Delhi**. By leveraging historical power consumption data and environmental factors, the model helps optimize energy distribution, reduce carbon footprints, and enhance grid stability.  

## **Features**  
✅ Predicts peak electricity demand for Delhi using ML algorithms  
✅ Uses historical data from **2013 to 2024** for training and testing  
✅ Implements **XGBoost** for high-accuracy forecasting  
✅ Supports energy optimization for **climate sustainability**  

## **Competition Track: Climate Change & Environmental Sustainability**  
Accurate energy forecasting minimizes waste, optimizes power generation, and integrates renewable energy sources, contributing to **sustainable urban energy management**.  

## **Tech Stack**  
- **Python**  
- **Pandas, NumPy** (Data Processing)  
- **XGBoost** (Machine Learning Model)  
- **Flask** (Web API for Predictions)  
- **OpenCV & TensorFlow** *(if used in future expansions)*  

## **Project Structure**  
```
Electric-power-demand-peaks-prediction/
│── templates/                 # Frontend templates (if applicable)
│── Powerdemand_actual_pridected_2013to2024.csv  # Dataset
│── dec7.csv                    # Processed dataset (latest data)
│── app.py                       # Flask app for predictions
│── model_xgbfinalft.pkl         # Trained ML model
│── imputerfinalft.pkl           # Preprocessing model
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

## **How to Run**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yug201/Electric-power-demand-peaks-prediction.git
   cd Electric-power-demand-peaks-prediction
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the Flask API:  
   ```bash
   python app.py
   ```  

## **Contributing**  
Feel free to fork the repository, raise issues, and submit pull requests.  


