# human-pose-tracking

## 環境構築

1. バージョン管理ツールの Rye をインストールする (https://rye-up.com/guide/installation/)
2. path を通す
3. リポジトリをクローンする
4. `rye sync`を実行

## 実行

### track (姿勢推定・人物追跡)

適宜、`config/config.ini` の値を変更する

```bash
$ rye run track
```

### CSV 変換 (convert)

track で生成されたフォルダの名前を `config/config.ini` 内の`convert.InputPath` に指定する

```bash
$ rye run convert
```

#### 信頼度出力 (confidence)

位置座標の信頼度を出力する

```bash
$ rye run confidence
```

### CSV 分割 (split)

```bash
$ rye run split 分割するファイルのパス 分割数(単位: 人数)
```
