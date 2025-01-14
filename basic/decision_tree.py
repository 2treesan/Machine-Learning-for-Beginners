from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

# 1. Dữ liệu
data = pd.DataFrame({
    'Màu sắc': ['Đỏ', 'Vàng', 'Đỏ', 'Đỏ', 'Vàng'],
    'Kích thước': ['Lớn', 'Nhỏ', 'Nhỏ', 'Lớn', 'Lớn'],
    'Loại trái cây': ['Táo', 'Chuối', 'Anh đào', 'Táo', 'Táo']
})

# 2. Mã hóa dữ liệu phân loại thành số
data_encoded = data.copy()
data_encoded['Màu sắc'] = data_encoded['Màu sắc'].map({'Đỏ': 0, 'Vàng': 1})
data_encoded['Kích thước'] = data_encoded['Kích thước'].map({'Lớn': 0, 'Nhỏ': 1})
data_encoded['Loại trái cây'] = data_encoded['Loại trái cây'].map({'Táo': 0, 'Chuối': 1, 'Anh đào': 2})

# 3. X và y
X = data_encoded[['Màu sắc', 'Kích thước']]
y = data_encoded['Loại trái cây']

# 4. Huấn luyện Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 5. Hiển thị cây quyết định
tree_rules = export_text(clf, feature_names=['Màu sắc', 'Kích thước'])
print(tree_rules)
