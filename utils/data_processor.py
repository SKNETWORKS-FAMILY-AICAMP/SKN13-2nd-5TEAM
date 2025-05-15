import os
import pandas as pd

def load_fitbit_data(base_path="data/raw/lifesnaps/rais_anonymized/csv_rais_anonymized"):
    daily_file = os.path.join(base_path, "daily_fitbit_sema_df_unprocessed.csv")

    if not os.path.exists(daily_file):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않음: {daily_file}")

    df = pd.read_csv(daily_file)

    # 날짜 컬럼 변환
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("❌ 'date' 컬럼이 존재하지 않습니다.")

    return df
