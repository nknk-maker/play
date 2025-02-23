#!/usr/bin/env python3
import sys

def unpack_packed_binary(input_file):
    """ pi.bin から 3.141592... の形の文字列を復元 """
    reverse_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

    with open(input_file, "rb") as f:
        packed_bytes = f.read()

    pi_text = []
    for byte in packed_bytes:
        high = (byte >> 4) & 0xF  # 上位4ビット
        low = byte & 0xF  # 下位4ビット
        if high < len(reverse_mapping):
            pi_text.append(reverse_mapping[high])
        if low < len(reverse_mapping):
            pi_text.append(reverse_mapping[low])

    return ''.join(pi_text)

def count_matching_digits(pi1, pi2):
    """ どこまで一致するかをカウント """
    match_count = 0
    for a, b in zip(pi1, pi2):
        if a == b:
            match_count += 1
        else:
            break
    return match_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} pi_bin_file out_txt_file".format(sys.argv[0]))
        sys.exit(1)

    pi_bin_file = sys.argv[1]
    out_txt_file = sys.argv[2]

    # pi.bin から復元
    restored_pi = unpack_packed_binary(pi_bin_file)

    # out.txt から読み込み
    with open(out_txt_file, "r", encoding="utf-8") as f:
        correct_pi = f.read().strip()

    # 一致する桁数を計算
    match_count = count_matching_digits(restored_pi, correct_pi)

    print(f"一致する桁数: {match_count} 桁")
