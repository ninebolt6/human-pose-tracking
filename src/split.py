import csv
from os import path
import sys


# CSVを1行ずつ読み込んで、指定した列数で分割する
def split(input_file_path: str, chunk_size_in_person_id: int):
    chunk_size = chunk_size_in_person_id * 2
    file_basename = path.splitext(path.basename(input_file_path))[0]

    # ファイルを開く
    with open(input_file_path, "r") as input_file:
        for row in csv.reader(input_file):
            row_size = len(row)

            # 1列目だけ抜き出す
            head_str = row[0]
            row = row[1:]

            for i in range(0, row_size - 1, chunk_size):
                # NOTE: x, yの2列で1人分のデータなので、ファイル名としては割る2されたものが正しい
                file_num = int(i / 2)
                with open(f"{file_basename}_{file_num}.csv", "a") as output_file:
                    output_row = row[i : i + chunk_size]
                    output_row.insert(0, head_str)

                    writer = csv.writer(output_file)
                    writer.writerow(output_row)


if __name__ == "__main__":
    # コマンドライン引数からファイル名と列数を取得する
    try:
        input_file_path = sys.argv[1]
        chunk_size_in_person_id = int(sys.argv[2])
    except:
        print("Usage: python split.py [input_file_path] [split person size]")
        sys.exit(1)

    split(input_file_path, chunk_size_in_person_id)
