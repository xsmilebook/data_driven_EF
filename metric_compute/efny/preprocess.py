import numpy as np
import pandas as pd

RT_MIN = 0.2
RT_MAX = 20.0


def to_numeric(series):
    s = pd.to_numeric(series, errors='coerce')
    return s


def clean_rt(rt):
    s = to_numeric(rt).dropna()
    s = s[(s >= RT_MIN) & (s <= RT_MAX)]
    if len(s) == 0:
        return s
    q1 = np.percentile(s, 25)
    q3 = np.percentile(s, 75)
    iqr = q3 - q1
    low = q1 - 3 * iqr
    high = q3 + 3 * iqr
    return s[(s >= low) & (s <= high)]


def determine_correct(df):
    if 'key' not in df.columns or 'answer' not in df.columns:
        return None
    a = df['answer'].astype(str).str.strip().str.lower()
    k = df['key'].astype(str).str.strip().str.lower()
    return k == a

