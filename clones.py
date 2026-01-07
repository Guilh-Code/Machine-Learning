# %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

df.columns

df["General Jedi encarregado"].unique()

# %%

features = ["Massa(em kilos)", "Estatura(cm)"]
target = "Status "

X = df[features]
y = df[target]

"""X = X.replace({
    "Yoda": 1,
    "Shaak Ti": 2,
    "Obi-Wan Kenobi": 3,
    "Aayla Secura": 4,
    "Mace Windu": 5
})"""

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X, y)

# %%

import matplotlib.pyplot as plt

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3)

# %%
