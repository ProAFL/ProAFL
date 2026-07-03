
from datetime import datetime

def get_time_str():
          
    now = datetime.now()
          
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def timestamp_to_hms(_timestamp)->str:
    hours = int(_timestamp // 3600)                   
    minutes = int((_timestamp % 3600) // 60)                     
    seconds = _timestamp % 60                               
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"

if __name__ == "__main__":
    get_time_str()