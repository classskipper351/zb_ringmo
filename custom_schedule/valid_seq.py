from collections import deque
import copy

def is_valid_state(sequences, nmb, nstages, log = False):
    assert len(sequences) == nstages #序列数量必须等于阶段数量
    
    for seq in sequences:
        try:
            assert len(seq) <= nmb *3 #每个序列的长度必须<操作数量
            for op in ['f','b','w']:
                assert seq.count(op) <= nmb #操作 {op} 的数量超过了限制
            F_stack = deque()
            B_stack = deque()

        
            for op in seq:
                if op == 'f':
                    F_stack.append(op)
                elif op == 'b':
                    B_stack.append(op)
                    F_stack.pop()
                elif op == 'w':
                    B_stack.pop()
        except Exception as e:
            if log:
                print(f"invalid sequnce {seq} on stage {sequences.index(seq)}")
            return False

    
    for stage in range(nstages-1):
        prev_seq = sequences[stage]
        succ_seq = sequences[stage+1]
        F_stack = deque()

        try:
            assert prev_seq.count('f') >= succ_seq.count('f') #"前一个阶段的 F 数量必须大于等于后一个阶段的 F 数量"
            assert prev_seq.count('b') <= succ_seq.count('b') #"前一个阶段的 B 数量必须小于等于后一个阶段的 B 数量"
                
        except Exception as e:
            if log:
                print( f"invalid sequnce {prev_seq} {succ_seq } on stage {sequences.index(prev_seq)} , {sequences.index(succ_seq)}")
            return False
    return True
             
def find_successors(sequences, nmb, nstages):
    if not is_valid_state(sequences, nmb, nstages):
        return []
    successors = []
    
    for stage in range(nstages):
        for op in ['f', 'b', 'w']:
            new_seq = [list(s) for s in sequences]
            new_seq[stage].append(op)
            if is_valid_state(new_seq, nmb, nstages):
                successors.append(new_seq)
                
    result = [
    [''.join(inner_list) for inner_list in middle_list]
    for middle_list in successors
    ]

    return result

def find_predecessors(sequences, nmb, nstages):
    if not is_valid_state(sequences, nmb, nstages):
        return []
    
    predecessors = []
    
    # 如果输入是字符串形式（如 ['ffbbw', 'fbfbww']），先转换成列表形式
    if isinstance(sequences[0], str):
        sequences = [list(seq) for seq in sequences]
    
    for stage in range(nstages):
        # 如果当前 stage 非空，尝试移除最后一个操作
        if sequences[stage]:  # 确保该 stage 有操作可移除
            new_seq = [list(s) for s in sequences]  # 深拷贝
            removed_op = new_seq[stage].pop()  # 移除最后一个操作
            if is_valid_state(new_seq, nmb, nstages):
                predecessors.append(new_seq)
    
    # 拼接成字符串形式返回
    result = [
        [''.join(inner_list) for inner_list in middle_list]
        for middle_list in predecessors
    ]
    
    return result

def calculate_mb_completed_stages(sequence, nmb, nstages):
    f_count=[seq.count('f') for seq in sequence]
    b_count=[seq.count('b') for seq in sequence]
    b_count.reverse()
    total_work = f_count + b_count
    
    def calculate_stages(total_check, nmb):
        res = [0] * (nmb*2)
        for i in range(len(total_check)):
            # 前 total_check[i] 名选手通过检查点 i
            for j in range(total_check[i]):
                res[j] += 1
        return res[:nmb]
    
    
    complete_satges = calculate_stages(total_work, nmb)
    #return [ nmb - x for x in complete_satges]
    return complete_satges

def huristic_estamate_cost(f,b,w,c ,nmb,nstages ,sequences):
    """
    估算成本函数
    huristic cost function
    """
    assert len(sequences) == nstages #序列数量必须等于阶段数量
    assert len(f) == nstages #阶段数量必须等于阶段数量
    assert len(b) == nstages #阶段数量必须等于阶段数量
    assert len(w) == nstages #阶段数量必须等于阶段数量
    
    f_count = 0
    b_count = 0
    w_count = [seq.count('w') for seq in sequences]
    
    
    from itertools import accumulate
    f_prefix_sum = list(accumulate(f))
    b_prefix_sum = list(accumulate(b))
    
    f_suffix_sum = [sum(f) - (f_prefix_sum[i] - f[i]) for i in range(nstages)]
    b_suffix_sum = [sum(b) - (b_prefix_sum[i] - b[i]) for i in range(nstages)]
    b_prefix_sum.reverse()
    
    
    fb_total_cost_list = [x+b_prefix_sum[0] for x in f_suffix_sum] + b_prefix_sum + [0]
    
    completed_stages  = calculate_mb_completed_stages(sequences, nmb, nstages)
    
    def relu(x):
        return max(0,x)
    
    def communication_counts(nstages, stage, cstage):
        assert cstage >= 0 and cstage <= 2*nstages 
        if cstage == 0 or cstage == 1:
            result =  2 * nstages - 2
        elif 1 < cstage <= nstages:
            result =  (2 * nstages - 2) - (cstage - 1)
        elif cstage == nstages + 1:
            result =  nstages - 1
        else:  # cstage > nstage + 1
            result =  max(nstages - 1 - (cstage - (nstages + 1)), 0)
            
        return relu(result - stage)
    
    stage_cost =[]
    for stage in range(nstages):
        
        local_cost_list =  [relu(x-(fb_total_cost_list[-stage-1])) for x in fb_total_cost_list]
        w_time = w[stage] *(nmb - w_count[stage])
        c_count_list = [communication_counts(nstages , stage ,cstage) for cstage in completed_stages]
        c_time = c * sum(c_count_list)
        local_cost = [local_cost_list[i] for i in completed_stages] 
        
        stage_cost.append(sum(local_cost) + w_time + c_time)
        
    return stage_cost
    
    
    
def get_local_time(f,b,w,c ,nmb,nstages ,sequences):
    
    assert len(sequences) == nstages #序列数量必须等于阶段数量
    assert len(f) == nstages #阶段数量必须等于阶段数量
    assert len(b) == nstages #阶段数量必须等于阶段数量
    assert len(w) == nstages
    assert is_valid_state(sequences, nmb, nstages)
    
    seq_len = [len(seq) for seq in sequences]
    seq_index = [0] * nstages
    curr_time = [0] * nstages
    f_queues = [deque() for _ in range(nstages)]
    b_queues = [deque() for _ in range(nstages)]
    
    def get_dep(stage ,op):
        assert op in ['f','b','w'] 
        if op == 'f':
            return stage-1 if stage > 0 else None
        elif op == 'b':
            return stage+1 if stage < nstages-1 else None
        return None
    
    while True:
        #try 1 
        for stage in range(nstages):
            
            if not seq_index[stage] < seq_len[stage]:
                continue
            op = sequences[stage][seq_index[stage]] 
            dep_stage = get_dep(stage ,op)
            
            if dep_stage is  None:
                if op == 'f':
                    curr_time[stage] += f[stage]
                    f_queues[stage].append(curr_time[stage])
                elif op == 'b':
                    curr_time[stage] += b[stage]
                    b_queues[stage].append(curr_time[stage])
                elif op == 'w':
                    curr_time[stage] += w[stage]   
                    
            else:
                if op == 'f':
                    if len(f_queues[dep_stage]) > 0:
                        dep_time = f_queues[dep_stage].popleft()
                        curr_time[stage] = max(curr_time[stage], dep_time+c) + f[stage]
                        f_queues[stage].append(curr_time[stage])
                    else:
                        continue
                elif op == 'b':
                    if len(b_queues[dep_stage]) > 0:
                        dep_time = b_queues[dep_stage].popleft()
                        curr_time[stage] = max(curr_time[stage], dep_time+c) + b[stage]
                        b_queues[stage].append(curr_time[stage])
                    else:
                        continue
            
            seq_index[stage] += 1
            
            
        if seq_index == seq_len:
            return curr_time
        
def is_ending_state(sequences, nmb, nstages):
    """
    检查序列是否为结束状态。
    
    参数:
        sequences: 包含 nstages 个操作序列的元组
        nmb: 每个操作类型（F, B, W）的数量
        nstages: 阶段数量
    返回:
        是否为结束状态（布尔值）
    """
    assert len(sequences) == nstages  # 序列数量必须等于阶段数量
    assert is_valid_state(sequences, nmb, nstages)  # 检查序列合法性
    for seq in sequences:
        if len(seq) != nmb * 3:
            return False
    return True

    
    
# 示例调用
if __name__ == "__main__":
    f=[1000,1000,1000,1000][:2]
    b=[100,100,100,100][:2]
    w=[10,10,10,10][:2]
    c=1
    nstages = 2
    nmb = 2
    seq = ['f','']


    # 检查序列合法性
    print("序列是否合法:", is_valid_state(seq, nmb, nstages,True))

    # 查找后继

    successors = find_successors(seq, nmb, nstages)
    print(f" {seq} 的合法后继状态:")
    for s in successors:
        print(s)
        

    # 查找前驱
    predecessors = find_predecessors(seq, nmb, nstages)
    print("合法前驱状态:")
    for p in predecessors:
        print(p)
        


    copmpleted_stages = calculate_mb_completed_stages(seq, nmb, nstages)
    print("每个微批次的完成的F+B总数:", copmpleted_stages)

    time = huristic_estamate_cost(f,b,w,c ,nmb,nstages,seq)
    print("估算每个stage运算完成的时间:", time)

    curr_time = get_local_time(f,b,w,c ,nmb,nstages,seq)
    print("每个阶段的当前时间:", curr_time)