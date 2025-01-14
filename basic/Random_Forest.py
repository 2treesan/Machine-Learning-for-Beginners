from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dữ liệu
iris = load_iris()
X, y = iris.data, iris.target

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tối ưu tham số
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Kết quả tối ưu
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy on CV:", grid_search.best_score_)

# Đánh giá trên tập kiểm tra
best_rf = grid_search.best_estimator_
accuracy = best_rf.score(X_test, y_test)
print("Accuracy on Test Data:", accuracy)
