import itertools
import random

def generate_valid_sequences(nmb):
    """
    生成所有长度为 3*nmb 的合法操作序列，满足任意前缀中 F >= B >= W。
    
    参数:
        nmb: 每个操作类型（F, B, W）的数量
    返回:
        合法序列的列表
    """
    def is_valid(seq):
        f_count, b_count, w_count = 0, 0, 0
        for op in seq:
            if op == 'F':
                f_count += 1
            elif op == 'B':
                b_count += 1
            elif op == 'W':
                w_count += 1
            if not (f_count >= b_count >= w_count):
                return False
        return True
    
    # 生成所有可能的 F, B, W 排列
    all_ops = ['F'] * nmb + ['B'] * nmb + ['W'] * nmb
    sequences = set(itertools.permutations(all_ops))  # 用 set 去重
    valid_sequences = [seq for seq in sequences if is_valid(seq)]
    return valid_sequences

def is_valid_state(sequences, nstages):
    """
    检查一组操作序列是否满足阶段间约束。
    
    参数:
        sequences: 包含 nstages 个操作序列的元组
        nstages: 阶段数量
    返回:
        是否合法（布尔值）
    """
    seq_len = len(sequences[0])
    for length in range(1, seq_len + 1):
        # 获取每个阶段在 length 长度下的前缀
        prefixes = [seq[:length] for seq in sequences]
        # 计算每个前缀的 F 和 B 数量
        f_counts = [prefix.count('F') for prefix in prefixes]
        b_counts = [prefix.count('B') for prefix in prefixes]
        # 检查 F 约束：F_current <= F_previous
        for stage in range(1, nstages):
            if f_counts[stage] > f_counts[stage - 1]:
                return False
        # 检查 B 约束：B_current <= B_next
        for stage in range(nstages - 1):
            if b_counts[stage] > b_counts[stage + 1]:
                return False
    return True

def count_and_sample_valid_states(nstages, nmb, sample_size=20):
    """
    计算合法状态的数量并打印示例。
    
    参数:
        nstages: 阶段数量
        nmb: 每个操作类型（F, B, W）的数量
        sample_size: 打印的示例状态数量
    """
    # 生成所有合法的单阶段序列
    valid_sequences = generate_valid_sequences(nmb)
    print(f"每个阶段的合法序列数量: {len(valid_sequences)}")
    
    # 生成所有阶段的合法序列组合
    all_combinations = itertools.product(valid_sequences, repeat=nstages)
    
    valid_states = []
    for combo in all_combinations:
        if is_valid_state(combo, nstages):
            valid_states.append(combo)
    
    total_valid = len(valid_states)
    print(f"合法状态总数: {total_valid}")
    
    # 打印示例状态
    print("其中示例状态:")
    if total_valid <= sample_size:
        for state in valid_states:
            print(state)
    else:
        sampled_states = random.sample(valid_states, sample_size)
        for state in sampled_states:
            print(state)

# 示例调用
nstages = 2  # 阶段数量
nmb = 2      # 微批次数量
squence = (('F', 'B','F' ), ('F', 'F', 'B', ))
print(is_valid_state(squence, nstages))