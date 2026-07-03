def get_cost_time(cost_timetamp)->str:
    hours = int(cost_timetamp // 3600)                   
    minutes = int((cost_timetamp % 3600) // 60)                     
    seconds = cost_timetamp % 60                               
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"