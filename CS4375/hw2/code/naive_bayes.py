import pandas as pd
import numpy as np


def predict(df, label, x):
    df_cols = df.drop(columns=[label]).columns
    p_y = df[label].value_counts(normalize=True)
    y_probs = []
    for y_name in p_y.index.tolist():
        p_xi = []
        for col in df_cols:
            n = len(df.loc[df[label] == y_name])
            p_xi.append(len((df.loc[df[label] == y_name]).loc[df[col] == x[col]]) / n)
        y_probs.append(p_y[y_name] * np.prod(p_xi))
    return p_y.index.tolist()[np.argmax(y_probs)]


def main():
    table_rows = [
        {"spam": "yes", "contains_win": "yes", "contains_free": "yes", "count": 40},
        {"spam": "yes", "contains_win": "yes", "contains_free": "no", "count": 25},
        {"spam": "yes", "contains_win": "no", "contains_free": "yes", "count": 30},
        {"spam": "yes", "contains_win": "no", "contains_free": "no", "count": 5},
        {"spam": "no", "contains_win": "yes", "contains_free": "yes", "count": 5},
        {"spam": "no", "contains_win": "yes", "contains_free": "no", "count": 15},
        {"spam": "no", "contains_win": "no", "contains_free": "yes", "count": 20},
        {"spam": "no", "contains_win": "no", "contains_free": "no", "count": 60},
    ]
    expanded_rows = []
    for row in table_rows:
        expanded_rows.extend(
            {
                "spam": row["spam"],
                "contains_win": row["contains_win"],
                "contains_free": row["contains_free"],
            }
            for _ in range(row["count"])
        )
    df = pd.DataFrame(expanded_rows)
    print(predict(df, "spam", {"contains_win": "no", "contains_free": "no"}))


if __name__ == "__main__":
    main()
