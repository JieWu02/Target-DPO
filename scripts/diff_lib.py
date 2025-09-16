import difflib
from transformers import AutoTokenizer
import torch
import json
import re
import random

def get_line_positions(text, tokenizer):
    """
    获取文本中每行在token序列中的精确位置
    """
    lines = text.splitlines()
    if not lines:
        return []

    # 先得到完整文本的token
    full_ids = tokenizer.encode(text)

    # 用来存储每行的token位置
    line_positions = []
    current_pos = 0

    # 对于每一行，找到其在token序列中的精确位置
    accumulated_text = ""
    for line in lines:
        # 计算当前行之前的文本长度
        line_start = len(accumulated_text)
        # 加上当前行
        accumulated_text += line + "\n"
        line_end = len(accumulated_text)

        # 将文本位置转换为token位置
        start_tokens = tokenizer.encode(text[:line_start])
        end_tokens = tokenizer.encode(text[:line_end])

        # 记录这一行对应的token范围
        line_positions.append((
            len(start_tokens),
            len(end_tokens)
        ))

    return line_positions


def get_precise_line_positions(text, tokenizer):
    """
    获取文本中每行在 token 序列中的精确位置，
    保留原始换行符的情况。
    """
    # 保留换行符
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    line_positions = []
    accumulated_text = ""
    for line in lines:
        line_start = len(accumulated_text)
        accumulated_text += line  # 直接使用保留换行符的行
        line_end = len(accumulated_text)

        start_tokens = tokenizer.encode(text[:line_start])
        end_tokens = tokenizer.encode(text[:line_end])
        line_positions.append((len(start_tokens), len(end_tokens)))
    return line_positions


def get_augmented_line_masks(chosen_ids, rejected_ids, tokenizer):
    """
    对比两段文本，找出不同的行，并标记对应的 token。
    对于 rejected 文本中的差异行，其前后随机 1～2 行也将被标记为 True。

    参数:
      chosen_ids: torch.Tensor 或列表，chosen 文本对应的 token id 序列。
      rejected_ids: torch.Tensor 或列表，rejected 文本对应的 token id 序列。
      tokenizer: 用于编码/解码的 tokenizer。

    返回:
      chosen_mask, rejected_mask: 分别对应 chosen 和 rejected 文本的 mask，
      它们均为与 token id 序列同长度的 BoolTensor，差异行（及其扩展行）对应位置为 True。
    """
    # 解码 token 序列得到文本
    chosen_text = tokenizer.decode(chosen_ids)
    rejected_text = tokenizer.decode(rejected_ids)

    # 清理特殊标记（如 <|im_end|>、<|endoftext|> 及其后续内容）
    # chosen_text = re.sub(r'<\|im_end\|>.*$', '', chosen_text)
    # rejected_text = re.sub(r'<\|im_end\|>.*$', '', rejected_text)
    # chosen_text = re.sub(r'<\|endoftext\|>.*$', '', chosen_text)
    # rejected_text = re.sub(r'<\|endoftext\|>.*$', '', rejected_text)

    eos_token = tokenizer.eos_token
    pattern = re.escape(eos_token) + '.*$'
    chosen_text = re.sub(pattern, '', chosen_text)
    rejected_text = re.sub(pattern, '', rejected_text)


    # 获取每行在 token 序列中的位置
    chosen_line_positions = get_precise_line_positions(chosen_text, tokenizer)
    rejected_line_positions = get_precise_line_positions(rejected_text, tokenizer)

    # 将文本按行分割
    chosen_lines = chosen_text.splitlines()
    rejected_lines = rejected_text.splitlines()

    # 利用 difflib 进行行级 diff 对比
    diff = list(difflib.ndiff(chosen_lines, rejected_lines))

    # 初始化 mask，默认所有位置为 False
    chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.bool)

    chosen_line = 0
    rejected_line = 0

    for d in diff:
        if d.startswith('  '):  # 两边相同的行
            chosen_line += 1
            rejected_line += 1
        elif d.startswith('- '):  # chosen 文本中有差异的行
            if chosen_line < len(chosen_line_positions):
                start, end = chosen_line_positions[chosen_line]
                chosen_mask[start:end] = True
            chosen_line += 1
        elif d.startswith('+ '):  # rejected 文本中有差异的行
            if rejected_line < len(rejected_line_positions):
                # 标记当前差异行对应的 tokens
                start, end = rejected_line_positions[rejected_line]
                rejected_mask[start:end] = True

                # 随机决定前后各扩展 1～2 行
                num_before = random.randint(1, 2)
                num_after = random.randint(1, 2)

                # 标记当前行之前的额外行（注意检查边界）
                for i in range(1, num_before + 1):
                    prev_index = rejected_line - i
                    if prev_index >= 0:
                        s, e = rejected_line_positions[prev_index]
                        rejected_mask[s:e] = True

                # 标记当前行之后的额外行
                for i in range(1, num_after + 1):
                    next_index = rejected_line + i
                    if next_index < len(rejected_line_positions):
                        s, e = rejected_line_positions[next_index]
                        rejected_mask[s:e] = True
            rejected_line += 1

    return chosen_mask, rejected_mask


def get_line_masks(chosen_ids, rejected_ids, tokenizer):
    """
    对比两段文本,找出不同的行,并标记对应的tokens
    """
    # 解码完整文本并清理特殊标记
    chosen_text = tokenizer.decode(chosen_ids)
    rejected_text = tokenizer.decode(rejected_ids)

    # chosen_text = re.sub(r'<\|im_end\|>.*$', '', chosen_text)
    # rejected_text = re.sub(r'<\|im_end\|>.*$', '', rejected_text)
    # chosen_text = re.sub(r'<\|endoftext\|>.*$', '', chosen_text)
    # rejected_text = re.sub(r'<\|endoftext\|>.*$', '', rejected_text)

    eos_token = tokenizer.eos_token
    pattern = re.escape(eos_token) + '.*$'
    chosen_text = re.sub(pattern, '', chosen_text)
    rejected_text = re.sub(pattern, '', rejected_text)

    # 获取每行的位置信息
    chosen_line_positions = get_precise_line_positions(chosen_text, tokenizer)
    rejected_line_positions = get_precise_line_positions(rejected_text, tokenizer)

    # 使用difflib找出差异
    chosen_lines = chosen_text.splitlines()
    rejected_lines = rejected_text.splitlines()
    diff = list(difflib.ndiff(chosen_lines, rejected_lines))

    # 初始化mask
    chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.bool)

    # 标记差异行对应的tokens
    chosen_line = 0
    rejected_line = 0

    for d in diff:
        if d.startswith('  '):  # 相同的行
            chosen_line += 1
            rejected_line += 1
        elif d.startswith('- '):  # chosen中的差异行
            if chosen_line < len(chosen_line_positions):
                start, end = chosen_line_positions[chosen_line]
                chosen_mask[start:end] = True
            chosen_line += 1
        elif d.startswith('+ '):  # rejected中的差异行
            if rejected_line < len(rejected_line_positions):
                start, end = rejected_line_positions[rejected_line]
                rejected_mask[start:end] = True
            rejected_line += 1

    # 打印标记结果进行验证
    # print("\nChosen text differences (marked with ###):")
    # for i, line in enumerate(chosen_lines):
    #     if i < len(chosen_line_positions):
    #         start, end = chosen_line_positions[i]
    #         prefix = "###" if chosen_mask[start:end].any() else "   "
    #         print(f"{prefix} {line}")
    #
    # print("\nRejected text differences (marked with ###):")
    # for i, line in enumerate(rejected_lines):
    #     if i < len(rejected_line_positions):
    #         start, end = rejected_line_positions[i]
    #         prefix = "###" if rejected_mask[start:end].any() else "   "
    #         print(f"{prefix} {line}")

    # 打印被标记的tokens
    # print("\nMarked tokens in chosen text:")
    # print(tokenizer.decode(chosen_ids[chosen_mask]))

    # print("\nMarked tokens in rejected text:")
    # print(tokenizer.decode(rejected_ids[rejected_mask]))

    return chosen_mask, rejected_mask

def get_prefix_suffix_masks(chosen_ids, rejected_ids, tokenizer):
    """
    基于共同前缀和后缀来标记差异部分。
    找到两个文本的最长公共前缀和最长公共后缀，中间的部分视为差异并标记为 True。
    
    参数:
      chosen_ids: torch.Tensor 或列表，chosen 文本对应的 token id 序列。
      rejected_ids: torch.Tensor 或列表，rejected 文本对应的 token id 序列。
      tokenizer: 用于编码/解码的 tokenizer。
    
    返回:
      chosen_mask, rejected_mask: 分别对应 chosen 和 rejected 文本的 mask，
      它们均为与 token id 序列同长度的 BoolTensor，差异部分对应位置为 True。
    """
    # 解码 token 序列得到文本
    chosen_text = tokenizer.decode(chosen_ids)
    rejected_text = tokenizer.decode(rejected_ids)
    
    # 清理特殊标记
    eos_token = tokenizer.eos_token
    pattern = re.escape(eos_token) + '.*$'
    chosen_text = re.sub(pattern, '', chosen_text)
    rejected_text = re.sub(pattern, '', rejected_text)
    
    # 按行分割文本
    chosen_lines = chosen_text.splitlines(keepends=True)
    rejected_lines = rejected_text.splitlines(keepends=True)
    
    # 找到最长公共前缀的行数
    prefix_lines = 0
    min_lines = min(len(chosen_lines), len(rejected_lines))
    
    for i in range(min_lines):
        if chosen_lines[i] == rejected_lines[i]:
            prefix_lines += 1
        else:
            break
    
    # 找到最长公共后缀的行数
    suffix_lines = 0
    for i in range(1, min_lines - prefix_lines + 1):
        if chosen_lines[-i] == rejected_lines[-i]:
            suffix_lines += 1
        else:
            break
    
    # 获取每行在 token 序列中的位置
    chosen_line_positions = get_precise_line_positions(chosen_text, tokenizer)
    rejected_line_positions = get_precise_line_positions(rejected_text, tokenizer)
    
    # 初始化 mask，默认所有位置为 False
    chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.bool)
    
    # 标记 chosen 文本中的差异部分
    if len(chosen_lines) > prefix_lines + suffix_lines:
        # 差异部分的起始行和结束行
        diff_start = prefix_lines
        diff_end = len(chosen_lines) - suffix_lines
        
        for line_idx in range(diff_start, diff_end):
            if line_idx < len(chosen_line_positions):
                start, end = chosen_line_positions[line_idx]
                chosen_mask[start:end] = True
    
    # 标记 rejected 文本中的差异部分
    if len(rejected_lines) > prefix_lines + suffix_lines:
        # 差异部分的起始行和结束行
        diff_start = prefix_lines
        diff_end = len(rejected_lines) - suffix_lines
        
        for line_idx in range(diff_start, diff_end):
            if line_idx < len(rejected_line_positions):
                start, end = rejected_line_positions[line_idx]
                rejected_mask[start:end] = True
    
    # 调试信息（可选）
    # print(f"\nCommon prefix lines: {prefix_lines}")
    # print(f"Common suffix lines: {suffix_lines}")
    # print(f"Chosen diff lines: {diff_start if len(chosen_lines) > prefix_lines + suffix_lines else 0} to {diff_end if len(chosen_lines) > prefix_lines + suffix_lines else 0}")
    # print(f"Rejected diff lines: {diff_start if len(rejected_lines) > prefix_lines + suffix_lines else 0} to {diff_end if len(rejected_lines) > prefix_lines + suffix_lines else 0}")
    
    return chosen_mask, rejected_mask

def get_token_level_masks(chosen_ids, rejected_ids, tokenizer):
    """
    优化版本：在行级差异的基础上，使用 token id 比较来找出差异 token。
    这个版本更快，因为直接比较 token ids 而不是字符。
    
    参数:
      chosen_ids: torch.Tensor 或列表，chosen 文本对应的 token id 序列。
      rejected_ids: torch.Tensor 或列表，rejected 文本对应的 token id 序列。
      tokenizer: 用于编码/解码的 tokenizer。
    
    返回:
      chosen_mask, rejected_mask: 分别对应 chosen 和 rejected 文本的 mask，
      它们均为与 token id 序列同长度的 BoolTensor，只有差异 token 对应位置为 True。
    """
    # 解码 token 序列得到文本
    chosen_text = tokenizer.decode(chosen_ids)
    rejected_text = tokenizer.decode(rejected_ids)
    
    # 清理特殊标记
    eos_token = tokenizer.eos_token
    pattern = re.escape(eos_token) + '.*$'
    chosen_text = re.sub(pattern, '', chosen_text)
    rejected_text = re.sub(pattern, '', rejected_text)
    
    # 获取每行在 token 序列中的位置
    chosen_line_positions = get_precise_line_positions(chosen_text, tokenizer)
    rejected_line_positions = get_precise_line_positions(rejected_text, tokenizer)
    
    # 将文本按行分割
    chosen_lines = chosen_text.splitlines()
    rejected_lines = rejected_text.splitlines()
    
    # 利用 difflib 进行行级 diff 对比
    diff = list(difflib.ndiff(chosen_lines, rejected_lines))
    
    # 初始化 mask
    chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.bool)
    
    # 处理差异
    chosen_line = 0
    rejected_line = 0
    
    i = 0
    while i < len(diff):
        if diff[i].startswith('  '):  # 相同的行
            chosen_line += 1
            rejected_line += 1
            i += 1
        elif diff[i].startswith('- '):  # chosen 中的行
            # 检查是否有对应的 rejected 行
            if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                # 这是一对差异行，进行 token 级别的比较
                if chosen_line < len(chosen_line_positions) and rejected_line < len(rejected_line_positions):
                    chosen_start, chosen_end = chosen_line_positions[chosen_line]
                    rejected_start, rejected_end = rejected_line_positions[rejected_line]
                    
                    # 获取这两行的 token ids
                    chosen_line_ids = chosen_ids[chosen_start:chosen_end].tolist() if torch.is_tensor(chosen_ids) else chosen_ids[chosen_start:chosen_end]
                    rejected_line_ids = rejected_ids[rejected_start:rejected_end].tolist() if torch.is_tensor(rejected_ids) else rejected_ids[rejected_start:rejected_end]
                    
                    # 使用 LCS 算法找出相同的 token
                    matcher = difflib.SequenceMatcher(None, chosen_line_ids, rejected_line_ids)
                    
                    # 标记 chosen 中的差异 token
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag in ['replace', 'delete']:
                            for idx in range(i1, i2):
                                chosen_mask[chosen_start + idx] = True
                        if tag in ['replace', 'insert']:
                            for idx in range(j1, j2):
                                rejected_mask[rejected_start + idx] = True
                
                chosen_line += 1
                rejected_line += 1
                i += 2
            else:
                # 整行只在 chosen 中存在
                if chosen_line < len(chosen_line_positions):
                    start, end = chosen_line_positions[chosen_line]
                    chosen_mask[start:end] = True
                chosen_line += 1
                i += 1
        elif diff[i].startswith('+ '):  # rejected 中的行
            # 整行只在 rejected 中存在
            if rejected_line < len(rejected_line_positions):
                start, end = rejected_line_positions[rejected_line]
                rejected_mask[start:end] = True
            rejected_line += 1
            i += 1
    
    return chosen_mask, rejected_mask

def main():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    # 读取JSON文件
    json_path = "../data/trl_dpo_data_lite.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理前10个数据
    for i, item in enumerate(data[150:200]):
        print(f"\n{'=' * 50}")
        print(f"Processing item {i + 1}/10")
        print('=' * 50)

        chosen_text = item['chosen']
        rejected_text = item['rejected']

        # 转换为token ids
        chosen_ids = torch.tensor(tokenizer.encode(chosen_text))
        rejected_ids = torch.tensor(tokenizer.encode(rejected_text))

        # 获取并打印差异
        chosen_mask, rejected_mask = get_line_masks(chosen_ids, rejected_ids, tokenizer)


if __name__ == "__main__":
    main()