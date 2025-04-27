import numpy as np
from itertools import chain


def get_windows(expert_num, ep_size, windows_size, balance=False, _set_=False):
    res = [
        list(range(expert_num // ep_size * i, expert_num // ep_size * (i + 1)))
        for i in range(ep_size)
    ]
    new_window = []
    if balance:
        return res
    else:
        for i in range(ep_size):
            new_window.append([*res[i], *res[(i + 1) % ep_size][:windows_size]])
        res = new_window
    if _set_:
        res = [set(i) for i in res]
    return res


def cpu_sort_(expert_token_size, token_size, expert_parallel_size, expert_windows):
    # expert_windows = [{0,1,2}, {0,1,2}, {1,2,3}, {2,3,4}, {3,4,5}, {4,5,6}, {5,6,7}, {5,6,7}]
    windows = expert_windows
    expert_windows = get_windows(
        len(expert_token_size), expert_parallel_size, expert_windows, False, True
    )
    expert_num = len(expert_token_size) // expert_parallel_size
    experts = [[] for _ in range(expert_parallel_size)]
    more_idx = []
    local_idx = [[] for _ in range(expert_parallel_size)]

    device_token_size = [
        token_size // expert_parallel_size for i in range(expert_parallel_size)
    ]
    expert_idx = 0
    tmp_size = 0
    index = 0
    while (
        index < expert_parallel_size
        and device_token_size[index % expert_parallel_size] > 0
    ):
        end = (index + 1) * expert_num
        start = index * expert_num
        if (
            sum(expert_token_size[start:end])
            <= device_token_size[index % expert_parallel_size]
        ):
            device_token_size[index % expert_parallel_size] -= sum(
                expert_token_size[start:end]
            )
            for j in range(start, end):
                if expert_token_size[j] > 0:
                    local_idx[index].append(expert_token_size[j])
                    experts[index].append(j)
                    expert_token_size[j] = 0
        else:
            sort_index = expert_token_size[start:end].argsort()
            for j in sort_index:
                if device_token_size[index] >= expert_token_size[start + j]:
                    local_idx[index].append(expert_token_size[start + j])
                    experts[index].append(start + j)
                    device_token_size[index] -= expert_token_size[start + j]
                    expert_token_size[start + j] = 0
                elif device_token_size[index] > 0:
                    local_idx[index].append(device_token_size[index])
                    experts[index].append(start + j)
                    expert_token_size[start + j] -= device_token_size[index]
                    device_token_size[index] = 0
                    more_idx.append([start + j, expert_token_size[start + j]])
                else:
                    more_idx.append([start + j, expert_token_size[start + j]])
        start = end % (expert_parallel_size * expert_num)
        end = (
            (start + windows) % (expert_parallel_size * expert_num)
            if start == 0
            else start + windows
        )
        if sum(expert_token_size[start:end]) <= device_token_size[index]:
            device_token_size[index] -= sum(expert_token_size[start:end])
            for j in range(start, end):
                if expert_token_size[j] > 0:
                    local_idx[index].append(expert_token_size[j])
                    experts[index].append(j)
                    expert_token_size[j] = 0
        else:
            sort_index = expert_token_size[start:end].argsort()
            for j in sort_index:
                if device_token_size[index] >= expert_token_size[start + j]:
                    local_idx[index].append(expert_token_size[start + j])
                    experts[index].append(start + j)
                    device_token_size[index] -= expert_token_size[start + j]
                    expert_token_size[start + j] = 0
                elif device_token_size[index] > 0:
                    local_idx[index].append(device_token_size[index])
                    experts[index].append(start + j)
                    expert_token_size[start + j] -= device_token_size[index]
                    device_token_size[index] = 0
        index += 1

    index = 0
    radio = token_size - sum(device_token_size)
    dts = []
    for idx, i in enumerate(device_token_size):
        if i > 0:
            dts.append([idx, i])

    def sort(dts, more_idx):
        device_token_size = sorted(dts, key=lambda b: b[1])
        more_idx = sorted(more_idx, key=lambda b: b[1])
        while device_token_size[-1][1] > more_idx[-1][1]:
            device_token_size[-1][1] -= more_idx[-1][1]
            local_idx[device_token_size[-1][0]].append(more_idx[-1][1])
            experts[device_token_size[-1][0]].append(more_idx[-1][0])
            more_idx.pop()
        if more_idx[-1][1] >= device_token_size[-1][1]:
            more_idx[-1][1] -= device_token_size[-1][1]
            experts[device_token_size[-1][0]].append(more_idx[-1][0])
            local_idx[device_token_size[-1][0]].append(device_token_size[-1][1])
            if more_idx[-1][1] == 0:
                more_idx.pop()
            device_token_size.pop()
        return device_token_size, more_idx

    while len(dts) > 0:
        dts, more_idx = sort(dts, more_idx)
    experts = [np.array(i) for i in experts]
    local_idx = [np.array(i) for i in local_idx]
    arg = [i.argsort() for i in experts]
    experts = [experts[index][arg[index]].tolist() for index in range(len(experts))]
    local_idx = [
        local_idx[index][arg[index]].tolist() for index in range(len(local_idx))
    ]
    return experts, local_idx


def sort_get_split_shape(expert_num, experts, local_idx, rank):
    split_shape = [[] for _ in range(len(expert_num))]
    for_which_rank = [[] for _ in range(len(expert_num))]
    input_output_shape = [[0 for _ in range(len(experts))] for _ in range(2)]
    input_shape, output_shape = input_output_shape
    shape = [{k: v for k, v in zip(i, j)} for i, j in zip(experts, local_idx)]
    jndex_ = [0 for _ in range(len(expert_num))]
    for index, i in enumerate(shape):
        for key in i:
            length = i[key]
            jndex = jndex_[key]
            expert = expert_num[key]
            tmp_jndex_ = 1
            while length > 0:
                tmp = expert[jndex]
                if tmp <= 0:
                    jndex += 1
                    continue
                if tmp <= length:
                    length -= tmp
                    min_ = tmp
                else:
                    expert[jndex] -= length
                    min_ = length
                    tmp_jndex_ = 0
                    length = 0
                if index == rank:
                    input_shape[jndex] += min_
                if jndex == rank:
                    split_shape[key].append(min_)
                    output_shape[index] += min_
                    for_which_rank[key].append(index)
                jndex += tmp_jndex_
            jndex_[key] = jndex
    for_which_rank = np.array(list(chain(*for_which_rank)))
    split_shape = list(chain(*split_shape))
    for_which_rank = for_which_rank.argsort()
    for_which_rank_ = for_which_rank.argsort()
    split_shape_1 = [split_shape[i] for i in for_which_rank]
    return (
        [split_shape, split_shape_1],
        [for_which_rank, for_which_rank_],
        input_output_shape,
    )


if __name__ == "__main__":
    a = get_windows(160, 16, 3)
    print(a)
