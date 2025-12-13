# atmaCup #21 ベースライン

このリポジトリ用に、コンペ提出形式どおりの **ごく簡単なベースライン** を用意しています。

## 構成

- `data/raw/input/atmaCup21__sample_submission.csv`
  - コンペ配布のサンプル提出。
- `src/baseline.py`
  - サンプル提出をそのまま `data/submission/submission_baseline.csv` としてコピーするだけのベースライン。

## 使い方

VS Code のターミナル、もしくはコマンドプロンプトでリポジトリ直下に移動して実行します。

```cmd
cd c:\Users\ishiz\pyworks\competitions\atma_21_elith
python -m src.baseline
```

実行すると `data/submission/submission_baseline.csv` が作成されます。そのままコンペサイトに提出可能です（内容は配布サンプルと同じです）。

今後は `src/baseline.py` 内のロジックを編集して、より強力な攻撃/防御プロンプトを自動生成する形に育てていく想定です。
