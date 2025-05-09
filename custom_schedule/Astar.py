import heapq
from  valid_seq import is_valid_state, find_successors, huristic_estamate_cost, get_local_time

def a_star_search(nmb, nstages, f, b, w, c):
    # 定义初始状态和结束判断函数
    start_state = [''] * nstages
    if not is_valid_state(start_state, nmb, nstages):
        return None

    # 优先级队列 (f, g, state)
    open_heap = []
    # 状态追踪字典 {state: (parent_state, g_score)}
    came_from = {}
    # 各状态的最佳g值记录
    g_scores = {}

    # 初始化第一个节点
    initial_g = max(get_local_time(f, b, w, c, nmb, nstages, start_state))
    initial_h = max(huristic_estamate_cost(f, b, w, c, nmb, nstages, start_state))
    heapq.heappush(open_heap, (initial_g + initial_h, initial_g, tuple(start_state)))
    g_scores[tuple(start_state)] = initial_g

    while open_heap:
        current_f, current_g, current_state = heapq.heappop(open_heap)
        current_state = list(current_state)  # 转换回列表便于处理

        # 结束状态判断
        if is_ending_state(current_state, nmb, nstages):
            return reconstruct_path(came_from, tuple(current_state))

        # 生成所有合法后继状态
        successors = find_successors(current_state, nmb, nstages)
        for successor in successors:
            successor_tuple = tuple(successor)
            
            # 计算实际成本
            time_cost = get_local_time(f, b, w, c, nmb, nstages, successor)
            successor_g = max(time_cost)
            
            # 计算启发估值
            h_cost = huristic_estamate_cost(f, b, w, c, nmb, nstages, successor)
            successor_h = max(h_cost)
            
            # 更新条件判断
            if successor_tuple not in g_scores or successor_g < g_scores.get(successor_tuple, float('inf')):
                came_from[successor_tuple] = tuple(current_state)
                g_scores[successor_tuple] = successor_g
                heapq.heappush(open_heap, (successor_g + successor_h, successor_g, successor_tuple))

    return None  # 无解

def reconstruct_path(came_from, end_state):
    path = []
    current_state = end_state
    while current_state in came_from:
        path.append(current_state)
        current_state = came_from[current_state]
    path.append(current_state)  # 加入初始状态
    return list(reversed(path))  # 返回正向路径

# 修改后的结束状态判断函数
def is_ending_state(sequences, nmb, nstages):
    if not is_valid_state(sequences, nmb, nstages):
        return False
    for seq in sequences:
        if len(seq) != nmb * 3:
            return False
    return True

# 使用示例
nmb = 8
nstages = 4
f = [8800,8800,8800,8800]
b = [6500]*4
w = [540]*4
c = 96

path = a_star_search(nmb, nstages, f, b, w, c)
if path:
    print("找到最优路径：")
    for state in path:
        print(state)
else:
    print("无解")