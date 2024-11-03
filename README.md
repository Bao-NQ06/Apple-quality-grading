# Apple Quality Classification Project

This project aims to classify the quality of apples as either "good" or "bad" using various machine learning models. The analysis includes logistic regression, ensemble methods, support vector machines, and a neural network. The models are trained and evaluated using a dataset of apple features, and hyperparameter tuning is performed using `GridSearchCV`.

## Project Structure
- **`apple_quality.csv`**: The dataset used for training and evaluating the models.
- **`main.ipynb`**: The main Jupyter notebook that contains the implementation of the models, data preprocessing, and performance evaluation.
- **`requirements.txt`**: List of all dependencies required to run the project.
- **`README.md`**: This file, providing an overview and setup instructions.

## Models Used
1. **Logistic Regression**
2. **Support Vector Machines (SVC, LinearSVC, NuSVC)**
3. **Ensemble Methods**:
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - Bagging
   - Extra Trees
   - Voting Classifier
   - Stacking Classifier
4. **Neural Network**: Implemented using PyTorch

## Performance Metrics
The models are evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **R² Score** (for regression-like evaluations)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/apple-quality-classification.git
cd apple-quality-classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing**: The data is cleaned, standardized, and split into training and testing sets.
2. **Model Training and Evaluation**: Run every cell in the **`main.ipynb`** file to understand clearly about the process of training and evaluating all models
3. **Hyperparameter Tuning**: GridSearchCV is used to optimize hyperparameters for SVM models.


## Results
The neural network achieved the highest performance with an accuracy of 95.25%, followed by NuSVC and SVC with accuracies of 92.62% and 92.00%, respectively. Ensemble methods like Random Forest and Stacking Classifier also performed well, demonstrating the effectiveness of combining multiple learners.

## Visualizations
* **Correlation Heatmap**: To understand the relationships between features.
* **Pairwise Scatter Plots**: To visualize the distribution and relationships of features.
* **Model Performance Table**: A comprehensive table summarizing all performance metrics for each model.

## Future Work
* **Explore Deep Learning**: Use convolutional neural networks (CNNs) for potential performance improvements.
* **Advanced Hyperparameter Tuning**: Implement techniques like Bayesian optimization.
* **Feature Engineering**: Experiment with new features or transformations to improve model accuracy.

