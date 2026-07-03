
from datetime import datetime

def get_time_str():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def timestamp_to_hms(_timestamp)->str:
    hours = int(_timestamp // 3600)  # 计算小时数
    minutes = int((_timestamp % 3600) // 60)  # 计算分钟数
    seconds = _timestamp % 60  # 计算剩余的秒数
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"

if __name__ == "__main__":
    get_time_str()