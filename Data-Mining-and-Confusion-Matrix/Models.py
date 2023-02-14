from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("sym_data.xlsx")
y = df.iloc[:,-1]
x = df.iloc[:,:-2]

model_names = [
    "Logistic Regression",
    "Gaussian NB",
    "KNN",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Network",
    "Gradient Boosting",
    "XGBoost",
    "Cat Boost"
]

classifier = [
    LogisticRegression(solver = "liblinear"),
    GaussianNB(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=1),
    SVC(gamma=0.1, C=5),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(max_iter=500),
    GradientBoostingClassifier(),
    XGBClassifier(),
    CatBoostClassifier()
]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

results = []

for name, model in zip(model_names, classifier):
    print(name)
    
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    results.append(score*100)
    
    print(score)
    
    plot_confusion_matrix(model, x_test, y_test)
    plt.title(name)
    
    filename = f'{name}.sav'
    pickle.dump(model, open(filename, 'wb'))
    
results_dict = {"Models":model_names,
                "Accuracy":results}

results_df = pd.DataFrame(results_dict)
print(results_df)

plt.figure()
sns.barplot(x= 'Accuracy', y = 'Models', data=results_df, color="b")
plt.xlabel('Accuracy %')
plt.show()