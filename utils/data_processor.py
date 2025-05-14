import os
import pandas as pd

def load_fitbit_data(base_path="data/raw/lifesnaps/rais_anonymized/csv_rais_anonymized"):
    """
    Lifesnaps Fitbit 데이터 로딩 함수 (daily granularity)
    :param base_path: CSV 파일들이 있는 디렉토리 경로
    :return: 일 단위 Fitbit DataFrame
    """
    daily_file = os.path.join(base_path, "daily_fitbit_sema_df_unprocessed.csv")

    if not os.path.exists(daily_file):
        raise FileNotFoundError(f"파일이 존재하지 않음: {daily_file}")

    df = pd.read_csv(daily_file)

    # 날짜 형식 변환
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("데이터에 'date' 컬럼이 없습니다.")

    # 누락된 주요 컬럼 확인
    required_columns = ['user_id', 'date', 'steps', 'calories', 'heartrate', 'sleep_minutes']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"⚠️ 누락된 예상 컬럼: {missing} (분석에 따라 수정 가능)")

    return df
