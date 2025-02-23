# 検証用のファイルにはhttps://data.mendeley.com/datasets/j6dp9rmdx2/1のpi3-100-million.txtを使用

import sys

def convert_pi_to_packed_binary(input_file, output_file): # 対応表：文字 -> 4ビット値 
    mapping = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10}

    # テキストから不要な空白や改行を除去
    with open(input_file, "r", encoding="utf-8") as f:
        pi_text = f.read().strip()

    # 各文字を対応する4ビット値に変換
    nibble_values = []
    for ch in pi_text:
        if ch in mapping:
            nibble_values.append(mapping[ch])
        else:
            # もし使用できない文字があればエラー
            raise ValueError("Unexpected character '{}' in file.".format(ch))

    # 4ビット値を2つずつまとめて1バイトにパックする
    packed_bytes = bytearray()

    i = 0
    while i < len(nibble_values):
        high = nibble_values[i]
        if i+1 < len(nibble_values):
            low = nibble_values[i+1]
        else:
            low = 0  # 奇数個の場合は下位4ビットを 0 でパディング
        byte_val = (high << 4) | low
        packed_bytes.append(byte_val)
        i += 2

    # バイナリファイルとして書き出す
    with open(output_file, "wb") as f:
        f.write(packed_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} input_file output_file".format(sys.argv[0]))
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print(input_file, output_file)
    convert_pi_to_packed_binary(input_file, output_file)