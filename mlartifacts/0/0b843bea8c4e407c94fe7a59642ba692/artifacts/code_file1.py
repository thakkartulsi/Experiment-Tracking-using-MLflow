import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

wine_df = load_wine()
X = wine_df.data
y = wine_df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2)

max_depth = 10
n_estimators = 10

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=2)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=wine_df.target_names, yticklabels=wine_df.target_names)
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.title('Confusion Matrix')

    plt.savefig('Confusion_Matrix.png')
    
    mlflow.log_artifact('Confusion_Matrix.png')
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'Author':'Tulsi', 'Project':'Wine Classification'})

    mlflow.sklearn.log_model(rf, 'Random Forest Classifier Model')


    print(accuracy)
