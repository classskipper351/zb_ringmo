
def find_first_difference(str1, str2):
    """
    比较两个字符串，返回第一个不相同的位置及其字符
    
    参数:
        str1 (str): 第一个字符串
        str2 (str): 第二个字符串
    
    返回:
        tuple: (位置索引, str1的字符, str2的字符)
              如果完全相同返回 None
              如果长度不同，超出部分用None表示
    """
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return (i, char1, char2)
    
    # 处理长度不同的情况
    if len(str1) != len(str2):
        shorter_len = min(len(str1), len(str2))
        if len(str1) > len(str2):
            return (shorter_len, str1[shorter_len], None)
        else:
            return (shorter_len, None, str2[shorter_len])
    
    return None  # 完全相同

   # (5, None, 'e')

schedule_not_uniformed = 'f0 sf0 f1 sf1 rb0 b0 f2 sf2 w0 rb1 b1 f3 sf3 w1 rb2 b2 f4 sf4 w2 rb3 b3 f5 sf5 w3 rb4 b4 f6 sf6 w4 rb5 b5 f7 sf7 w5 rb6 b6 w6 rb7 b7 w7\
rf0 rf1 f0 b0 sb0 f1 rf2 b1 sb1 f2 rf3 b2 sb2 f3 rf4 b3 sb3 w0 f4 rf5 b4 sb4 w1 f5 rf6 b5 sb5 w2 f6 rf7 b6 sb6 w3 f7 b7 sb7 w4 w5 w6 w7'

schedule_uniformed = 'f0 sf0 f1 sf1 rb0 b0 rb1 f2 sf2 b1 f3 sf3 rb2 w0 b2 rb3 f4 sf4 b3 f5 sf5 rb4 w1 b4 rb5 f6 sf6 b5 w2 rb6 f7 sf7 w3 w4 w5 b6 w6 rb7 b7 w7\
rf0 rf1 f0 b0 sb0 f1 b1 sb1 rf2 w0 f2 rf3 b2 sb2 f3 b3 sb3 rf4 w1 f4 rf5 b4 sb4 f5 b5 sb5 rf6 w2 f6 b6 sb6 rf7 w3 f7 b7 sb7 w4 w5 w6 w7'

print(find_first_difference(schedule_not_uniformed, schedule_uniformed)) 
