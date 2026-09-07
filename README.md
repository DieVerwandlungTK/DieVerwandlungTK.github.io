# Research homepage

自己紹介・研究紹介・Publications・Presentations・Links を持つ静的サイトです。
背景は **vgpu 0.4.0 / WebGPU** で、同梱の水分子動力学軌跡を再生します。
酸素と水素を描き、水素結合の生成と切断が見えます。
個人情報・業績・リンクは placeholder です。

## ローカルで表示

Node.js 22.12以上（または24系）を使用してください。

```sh
npm ci
npm run dev
```

表示される localhost のURLをブラウザで開きます。
WebGPU非対応・初期化失敗・データ取得失敗時も、本文と静止画が表示されます。
動きを減らすOS設定では自動再生しません。再生ボタンで明示的に開始できます。
再生は1倍速で約30秒（軌跡40 ps）で停止します。0.5・1・2・4倍速を選べます。
逆再生は行いません。

```sh
npm test
npm run build
npm run preview
```

ブラウザ検証は `npx playwright install chromium` のあと、
`npm run build && npm run test:browser` で実行します。GPU描画のテストは
WebGPUが使える環境が必要です（Chromiumの通常ヘッドレスモードを使用）。

## 編集する場所

- `index.html`: 氏名、所属、紹介文、研究内容、論文、学会発表、リンク、ページタイトル。
- `src/style.css`: 色、レイアウト、文字サイズ。
- `src/background.ts` / `src/scene.ts`: 分子の配置・投影・結合判定。
- `src/particles.wgsl` / `src/bonds.wgsl`: 原子と結合の描画。
- `public/data/water.json` + `water.bin`: 軌跡。読み込み仕様は `src/trajectory.ts`、
  データ形式は `docs/explicit-water-simulation.md` にあります。
- `public/molecules.svg`: WebGPUが使えない場合の初期フレーム。

Linksの「未設定」は意図的にリンクにしていません。設定時は
`<div class="profile-link">...</div>` を実URLの `<a class="profile-link" href="…">...</a>`
に変更してください。研究テーマと紹介文もサンプルなので公開前に置き換えます。
Google Fontsからフォントを読み込みます。ネットワークが使えない場合はローカルの
serif/sans-serifフォントに切り替わります。

## GitHub Pages

1. このフォルダを自身のGitHubリポジトリにpushします。
2. リポジトリの **Settings → Pages → Build and deployment → Source** で
   **GitHub Actions** を選びます。
3. `main` へのpush、またはActions画面から `Deploy homepage to GitHub Pages` を
   実行すると、テスト・ビルド後に公開されます。

Viteは `base: './'` でビルドするため、`https://USERNAME.github.io/REPOSITORY/`
のようなサブディレクトリにも対応します。Pythonはサイトのビルドに不要です。
この作業環境にはGitHubリモートが未設定のため、公開はまだ行っていません。

## サンプル計算の内容と再生成

剛体 **TIP4P-Ew** 水64分子の軌跡です。酸素と2つの水素を描画し、電荷サイト
（massless virtual site）は描画しません。分子内O-H結合は拘束されており、加熱でも
解離しません。破線は軌跡から幾何学的に判定した水素結合（O···O ≤ 3.5 Å、
O-H···O ≥ 150°、最短像距離）で、追加の力ではありません。表示は周期境界のバルク
水を球状に切り出した窓で、液滴や表面ではありません。

原論文: H. W. Horn *et al.*, *J. Chem. Phys.* **120**, 9665–9678 (2004),
[doi:10.1063/1.1683075](https://doi.org/10.1063/1.1683075)。
vgpuはMITライセンスです（`node_modules/vgpu/LICENSE`）。

| 条件 | 値 |
|---|---|
| 初期構造 | 陽子無秩序な氷Ic（2×2×2ダイヤモンド格子）、64分子 |
| 箱 | 12.7 Å立方、周期境界、定積 |
| 力場 | OpenMMの`tip4pew.xml`、PME、実空間カットオフ6 Å |
| 積分 | Langevin-middle、2 fs刻み、摩擦1 ps⁻¹、剛体水 |
| 平衡化 | エネルギー最小化後、180 Kで10 ps |
| 記録 | 180 Kで5 ps、20 psで450 Kへ昇温、450 Kで15 ps |
| 保存 | 20 fsごと、2001フレーム、40 ps |
| 乱数 | seed 22039（陽子配置・サーモスタット） |

Python 3.13 と [uv](https://docs.astral.sh/uv/) を使う場合:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r scripts/requirements.txt
.venv/bin/python scripts/generate_explicit_water.py
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv/bin/python scripts/generate_static.py
```

生成器は座標の有限性、剛体形状、周期変位、熱的変化、構造変化、水素結合の
入れ替わりを確認してから既存データを置き換えます。検証結果と限界は
`docs/explicit-water-simulation.md` にまとめています。急速加熱・小規模・定積の
計算であり、平衡融点や実験条件の再現を検証した研究データではありません。
画面の温度はサーモスタットの設定値で、瞬間運動温度はJSONに記録しています。

軌跡を差し替える場合は同じ形式（`version: 2`）にします。`water.json` に
フレームごとの `timePs`・`temperature` を、`water.bin` にリトルエンディアン
float32でフレーム→分子→原子（O, H, H）→xyzの順に座標をÅで格納します。
酸素は `[0, box)` に折り返し、水素は分子ごとに酸素とつなげたまま保存します
（箱の外に出てよい）。周期境界での最短像補間を使うため、フレーム間の酸素変位は
各軸で半ボックス未満になる頻度で保存してください。分子数・温度・説明文と
静止画（`scripts/generate_static.py`）も併せて更新します。
