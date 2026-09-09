# Research homepage

自己紹介・研究紹介・Publications・Presentations・Links を持つ静的サイトです。
別ページの [Playground](playground.html) は **vgpu 0.4.0 / WebGPU** で、剛体TIP4P-Ew水分子動力学をブラウザ上でリアルタイムに計算・描画します。
録画済みの軌跡を再生しているのではありません。酸素と水素を描き、温度・密度・分子数・計算速度を
訪問者が操作でき、水素結合の生成と切断が見えます。
[ブラウン運動と SDE](brownian.html) では、2次元の理想気体と追跡粒子の弾性衝突を、
推定した拡散係数を使う過減衰 SDE と比較できます。Canvas 2D で動作し、WebGPU は不要です。
温度・数密度は無次元量で、64個の独立した系を計算します。準備運転20、係数推定80、比較60の
順に進み、MSD・変位分布を表示します。推定区間の後半の傾きから D を求め、比較中は固定します。
条件変更で推定をやり直し、初めからボタンで同じ乱数種の結果を再現できます。
条件、推定区間と比較区間の数値、最終変位は JSON として保存できます。
有限時間・有限サイズ・刻み幅の近似を含む教材用モデルで、水の物性値には対応しません。

個人情報・業績・リンクは placeholder です。

## ローカルで表示

Node.js 22.12以上（または24系）を使用してください。

```sh
npm ci
npm run dev
```

表示される localhost のURLをブラウザで開きます。
トップページではシミュレーションを読み込まず、ナビゲーションの Playground から実験ページへ移動できます。
WebGPU非対応・初期化失敗・データ取得失敗時も、本文と静止画が表示されます。
動きを減らすOS設定では自動計算を開始しません。再開ボタンで明示的に開始できます。
計算は氷Icの初期配置から継続的に進み、あらかじめ決まった長さや終了時刻はありません。
温度（150〜500 K）・密度（60〜140%）・分子数（64／216／512）・計算速度
（0.5・1・2・4倍、いずれも実時間換算ではなく1フレームあたりのステップ数）を操作できます。
リセットボタンは氷Icの初期配置に計算をやり直します。

パネル内の速度分布は、分子の重心の並進速度を12区間のヒストグラムで表示し、
実測の並進運動温度に対応するMaxwell–Boltzmann曲線を重ねます。横軸0〜18 Å/ps、
縦軸0〜0.26 ps/Åは固定です。200 msごとに計測し、重み1/3の指数移動平均で
揺らぎを抑えます。18 Å/psを超えた速度は最後の区間に集計します。
曲線との一致はLangevin熱浴の正準サンプリングの確認であり、水の相互作用の検証ではありません。
動きを減らす設定では初期分布だけを描画し、明示的に再開するまで更新しません。

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
- `src/style.css`: ホームページと共通の色、レイアウト、文字サイズ。
- `playground.html` / `src/playground.css`: 実験ページの説明、操作パネル、レイアウト。
- `src/playground.ts`: 実験ページの起動・操作。計算と描画は独立したモジュールに保ち、将来の配置変更にも再利用できます。
- `brownian.html` / `src/brownian.css` / `src/brownian.ts`: ブラウン運動比較の画面・操作・保存。
- `src/brownian-model.ts` / `src/brownian-experiment.ts`: 衝突・SDE・推定と比較の進行。
- `src/brownian-charts.ts`: 代表軌跡、MSD、変位分布の描画。
- `src/speed-distribution.ts` / `src/distribution-chart.ts`: 並進速度の集計・理論分布・SVGグラフ。
- `src/background.ts` / `src/scene.ts`: 分子の配置・投影・結合判定。
- `src/simulation.ts` / `src/simulation.wgsl`: ブラウザ上で動くTIP4P-Ew剛体水の力場と積分。
- `src/particles.wgsl` / `src/bonds.wgsl`: 原子と結合の描画。
- `public/data/ice-64.bin` / `ice-216.bin` / `ice-512.bin`: 各分子数の氷Ic初期配置。
  データ形式は `docs/explicit-water-simulation.md` にあります。
- `public/molecules.svg`: WebGPUが使えない場合の静止画（216分子の初期配置）。

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

## サンプル計算の内容と再生成

剛体 **TIP4P-Ew** 水分子を、`src/simulation.wgsl` の力場でブラウザ上でその場計算します。
録画済みの軌跡を再生しているのではありません。酸素と2つの水素を描画し、電荷サイト
（massless virtual site）は描画しません。分子内O-H結合は拘束されており、加熱でも
解離しません。破線はその場で幾何学的に判定した水素結合（O···O ≤ 3.5 Å、
O-H···O ≥ 150°、最短像距離）で、追加の力ではありません。表示は周期境界のバルク
水を球状に切り出した窓で、液滴や表面ではありません。

訪問者は温度（150〜500 K）・密度（60〜140%）・分子数（64／216／512）・計算速度
（0.5・1・2・4倍、1フレームあたりのステップ数）を操作できます。一時停止／再開ボタンと、
氷Icの初期配置に戻すリセットボタンがあります。

原論文: H. W. Horn *et al.*, *J. Chem. Phys.* **120**, 9665–9678 (2004),
[doi:10.1063/1.1683075](https://doi.org/10.1063/1.1683075)。
vgpuはMITライセンスです（`node_modules/vgpu/LICENSE`）。

| 分子数 | 箱（立方） | カットオフ |
|---|---|---|
| 64 | 12.7 Å | 6.2 Å |
| 216 | 19.05 Å | 9 Å |
| 512 | 25.4 Å | 9 Å |

いずれも初期構造は陽子無秩序な氷Ic、周期境界、定積です。積分は2 fs刻みのBAOAB
Langevin、摩擦5 ps⁻¹（この摩擦が動力学を減衰させています）。乱数はseed 22039
（陽子配置・サーモスタット）。

**近似であることを明記します。** 静電相互作用はEwald合計（PME）ではなく、
カットオフ付きOnsager反応場（`min(9, 0.49×箱)` Å、導体境界条件）で近似しています。
GPU上の計算はf32（単精度）で行われるため、エネルギーは厳密には保存されません。
サーモスタットの摩擦5 ps⁻¹は動力学そのものを減衰させます。**そして、冷却しても
再凍結はしません。** 結晶核形成はこの系のサイズと時間スケールをはるかに超えるため、
冷却するとアモルファス（非晶質）固体になるだけです。リセットボタンは氷Icの初期配置に
計算をやり直すのであって、凍結を起こすものではありません。

Python 3.13 と [uv](https://docs.astral.sh/uv/) を使う場合:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r scripts/requirements.txt
.venv/bin/python scripts/generate_explicit_water.py     # 氷の初期配置と検証用軌跡
.venv/bin/python scripts/reference_forces.py            # GPUテスト用の力のフィクスチャ
.venv/bin/python scripts/generate_static.py             # 静止画フォールバック
.venv/bin/python scripts/generate_reference_rdf.py ice     # 300 K・過熱氷Icの参照RDF
.venv/bin/python scripts/generate_reference_rdf.py fluid   # 520 K・流体の参照RDF
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

生成器は座標の有限性、剛体形状、周期変位、熱的変化、構造変化、水素結合の
入れ替わりを確認してから既存データを置き換えます。検証結果と限界は
`docs/explicit-water-simulation.md` にまとめています。

`reference/water.json` と `reference/water.bin` にあるOpenMMの軌跡は、**もう画面には
表示されません**。急速加熱・小規模・定積の参照計算で、ブラウザ内シミュレーションの
力場・積分を検証するために残しています（`scripts/test_explicit_water.py`、
`scripts/reference_forces.py`）。平衡融点や実験条件の再現を検証した研究データでは
ありません。力は `scripts/reference_forces.py` と、構造は `tests/fixtures/reference-rdf-ice-300k.json`
（300 K・過熱された結晶氷Ic）と `tests/fixtures/reference-rdf-fluid-520k.json`
（520 K・乱れた流体）の2状態と、それぞれ比較検証していますが、**常温での平衡液体水の
構造は検証していません**——このタイムスケールでは300 Kの試料はブラウザでもOpenMMでも
融解しないためです。
