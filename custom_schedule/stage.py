def calculate_stages(total_check, nmb):
    res = [0] * nmb
    for i in range(len(total_check)):
        # 前 total_check[i] 名选手通过检查点 i
        for j in range(total_check[i]):
            res[j] += 1
    return res

# 测试
total_check = [3, 0, 0, 0]
nmb = 4
print(calculate_stages(total_check, nmb))  # 输出: [3, 1, 0, 0]