import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

with open("groceries.csv", "r", encoding="utf-8") as f:
    dataset = [line.strip().split(",") for line in f if line.strip()]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.25, use_colnames=True)
print("Frequent Itemsets with min_support = 0.25:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules (Items brought together):")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
