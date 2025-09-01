import pandas as pd
from pathlib import Path
from typing import List, Tuple

# ===== הגדרות =====
FILES: List[str] = [
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset1.csv",
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset2.csv",
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset3.csv",
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset4.csv",
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset5.csv",
    r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\labeled_dataset6.csv"
]
OUT_CSV = r"C:\Users\veron\OneDrive - Yezreel Valley College\Suicide_Behavior_Detection_Project\data\processed\final_balanced_dataset2.csv"
RANDOM_STATE = 42
TARGET_PER_CLASS = 900

def load_and_check(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Post_Id", "Text", "Lable"}
    if set(df.columns) & expected != expected:
        raise KeyError(f"{path} חייב לכלול עמודות בדיוק: {sorted(expected)}. יש: {list(df.columns)}")
    # נוודא טיפוסים
    df["Text"] = df["Text"].fillna("").astype(str).str.strip()
    df["Lable"] = pd.to_numeric(df["Lable"], errors="coerce").astype("Int64")
    if df["Lable"].isna().any():
        raise ValueError(f"יש NaN בעמודת Lable ב-{path}")
    if not set(df["Lable"].unique()).issubset({0, 1}):
        raise ValueError(f"עמודת Lable ב-{path} צריכה להיות 0/1 בלבד")
    df["Post_Id"] = df["Post_Id"].astype(str)
    # מסננים טקסטים ריקים לגמרי
    return df[df["Text"].str.len() > 0].copy()

def build_balanced(files: List[str],
                   target_per_class: int = TARGET_PER_CLASS,
                   random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, Tuple[int,int,int]]:
    parts = [load_and_check(p) for p in files if Path(p).exists()]
    if not parts:
        raise FileNotFoundError("לא נטענו קבצים. בדקי את הנתיבים.")

    df = pd.concat(parts, ignore_index=True)

    # הסרת כפילויות:
    # 1) כפילות מלאה (אותו Post_Id,Text,Lable)
    df = df.drop_duplicates(subset=["Post_Id", "Text", "Lable"], keep="first")
    # 2) אם אותו טקסט עם אותו לייבל מופיע כמה פעמים עם Post_Id שונה — נשאיר אחד
    df = df.drop_duplicates(subset=["Text", "Lable"], keep="first").reset_index(drop=True)

    have0 = int((df["Lable"] == 0).sum())
    have1 = int((df["Lable"] == 1).sum())

    if have0 == 0 or have1 == 0:
        raise ValueError(f"לא ניתן לאזן ללא שכפול: Lable=0={have0}, Lable=1={have1}")

    n = min(target_per_class, have0, have1)  # בלי oversample
    pos = df[df["Lable"] == 1].sample(n=n, random_state=random_state)
    neg = df[df["Lable"] == 0].sample(n=n, random_state=random_state)

    balanced = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced = balanced[["Post_Id", "Text", "Lable"]]
    return balanced, (have0, have1, n)

if __name__ == "__main__":
    balanced_df, (have0, have1, used_per_class) = build_balanced(FILES, TARGET_PER_CLASS, RANDOM_STATE)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"נשמר קובץ מאוזן (ללא oversample) → {OUT_CSV}")
    print(f"זמינות לפני איזון: Lable=0: {have0} | Lable=1: {have1}")
    print(f"בפועל נבחר: {used_per_class} לכל מחלקה → סה״כ {2*used_per_class}")