#!/usr/bin/env python3
import sys

def count_matching_digits(file1, file2):
    """ 2つのテキストファイルを比較し、一致する桁数をカウント """
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        text1 = f1.read().strip()
        text2 = f2.read().strip()
    
    match_count = 0
    for a, b in zip(text1, text2):
        if a == b:
            match_count += 1
        else:
            break
    
    return match_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} file1 file2".format(sys.argv[0]))
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    match_count = count_matching_digits(file1, file2)
    print(f"一致する桁数: {match_count} 桁")