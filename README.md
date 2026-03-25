# titanic_survival_survey

Built a Machine Learning classification model to predict passenger survival on the Titanic using the classic Seaborn Titanic dataset.
The project covers the complete ML pipeline — data cleaning, feature engineering, label encoding, model training, and performance
evaluation ,comparing multiple classification algorithms to identify the best performing model.

1. Logistic Regression
A linear classification algorithm used as the baseline model. Predicts survival probability using a sigmoid function and outputs binary
classification — survived or not survived. Evaluated using confusion matrix and full classification report with precision, recall, and F1-score.

2. K-Nearest Neighbors (KNN)
A distance-based algorithm trained with k=5 neighbors on StandardScaler normalized features. Classifies each passenger based on the majority
class among its 5 nearest data points in feature space.

4. Gaussian Naive Bayes
A probabilistic classifier based on Bayes theorem assuming feature independence with Gaussian distribution. Trained on raw features and evaluated
using accuracy score, confusion matrix, and classification report.

5. Decision Tree Classifier
A tree-based non-linear classification model trained with random_state=42 for reproducibility. Splits data based on feature thresholds to build
 a decision tree that predicts survival outcomes.
