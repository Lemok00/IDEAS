def time_change(time_init):   #定义将秒转换为时分秒格式的函数
    time_list = []
    if time_init/3600 > 1:
        time_h = int(time_init/3600)
        time_m = int((time_init-time_h*3600) / 60)
        time_s = int(time_init - time_h * 3600 - time_m * 60)
        time_list.append(str(time_h))
        time_list.append('h ')
        time_list.append(str(time_m))
        time_list.append('m ')

    elif time_init/60 > 1:
        time_m = int(time_init/60)
        time_s = int(time_init - time_m * 60)
        time_list.append(str(time_m))
        time_list.append('m ')
    else:
        time_s = int(time_init)

    time_list.append(str(time_s))
    time_list.append('s')
    time_str = ''.join(time_list)
    return time_str