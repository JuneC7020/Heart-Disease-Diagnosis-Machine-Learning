# Heart Disease Diagnosis Machine Learning

This project implements machine learning algorithms to predict the presence of heart disease based on patient data. The dataset used is sourced from the UCI Machine Learning Repository, and models like Neural Networks, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), and AdaBoost are used for classification tasks. Various data preprocessing techniques, such as SMOTE for handling class imbalance, and dimensionality reduction techniques, such as PCA and t-SNE, are also applied.

## Project Structure

- **ML_A1Minjun_heart_disease.ipynb**: Jupyter notebook containing the entire pipeline for heart disease diagnosis prediction, including data preprocessing, model training, and evaluation.
- **data/**: Directory to store the Heart Disease dataset (download instructions below).
- **results/**: Directory to store the results of the experiments, such as model performance metrics.

## Dataset

The Heart Disease dataset is available on the UCI Machine Learning Repository. You can download it from the following link:
- [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

Place the dataset in the `data/` folder before running the notebook.

## Requirements

### Libraries

The following Python libraries are required to run the project:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn (optional for visualizations)
- Imbalanced-learn (for SMOTE)

You can install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn
