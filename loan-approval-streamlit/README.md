# Loan Approval Prediction Streamlit Application

This project is a Streamlit application for predicting loan approvals based on various features. The application utilizes a machine learning model trained on a dataset of loan applications. 

## Project Structure

- **app.py**: Main entry point for the Streamlit application. It loads the model, processes user inputs, and displays predictions and visualizations.
- **requirements.txt**: Lists the dependencies required for the project, including Streamlit, pandas, numpy, scikit-learn, and other necessary libraries.
- **.gitignore**: Specifies files and directories to be ignored by Git, such as compiled Python files and virtual environments.
- **data/loan_approval.csv**: Dataset used for the loan approval analysis and model training.
- **notebooks/eda4.ipynb**: Jupyter Notebook containing exploratory data analysis and modeling code that serves as the basis for the Streamlit application.
- **src/**: Contains various Python modules for data processing, feature engineering, visualization, model training, and prediction.
  - **data_processing.py**: Functions for data cleaning and preprocessing.
  - **features.py**: Functions for feature engineering.
  - **visualization.py**: Functions for visualizing data and model results.
  - **models.py**: Model training and evaluation functions.
  - **predict.py**: Functions for making predictions using the trained model.
  - **utils.py**: Utility functions for use across different modules.
- **models/loan_model.joblib**: Serialized trained model ready for predictions.
- **tests/**: Contains unit tests for the project.
  - **test_data_processing.py**: Unit tests for data processing functions.
  - **test_models.py**: Unit tests for model training and prediction functions.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd loan-approval-streamlit
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit application, execute the following command in your terminal:
```
streamlit run app.py
```

This will start the Streamlit server and open the application in your web browser.

## Usage

- Input the required features in the provided fields.
- Click on the "Predict" button to see the loan approval prediction.
- Visualizations will be displayed based on the input data and model predictions.

## License

This project is licensed under the MIT License.