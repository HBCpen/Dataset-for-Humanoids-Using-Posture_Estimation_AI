# ヒューマノイドロボットのための学習データセットと身体性AIにおける技術的課題：包括的調査報告書

## 序論：身体性AIにおけるデータ駆動型パラダイムの到来とヒューマノイドの特異性

人工知能（AI）研究の歴史において、モラベックのパラドックスが示唆してきたように、高度な推論よりも物理的な相互作用の習得が困難であるという事実は、長らくロボティクス分野の進展を阻む壁であった。しかし、近年の大規模言語モデル（LLM）や視覚言語モデル（VLM）の爆発的な成功は、ロボット制御、特にヒューマノイド（人型ロボット）の研究開発においても、従来のモデルベース制御からデータ駆動型の学習アプローチへの劇的なパラダイムシフトを引き起こしている。この潮流の中で、ロボットが物理世界を理解し、操作するための「**身体性AI（Embodied AI）**」の実現に向けた競争が激化しており、その成否を握る最大の鍵が「学習データ」の質と量であることは論を待たない。

汎用的なヒューマノイドロボットの開発は、単なる産業用アームロボットの延長線上にはない。ヒューマノイドは、二足歩行による移動（Locomotion）と、多自由度ハンドによる物体操作（Manipulation）を同時に、かつ協調して行う「**全身操作（Loco-manipulation）**」を必要とする。この複雑性は、固定されたベースを持つアームロボットとは比較にならないほどの高次元な状態空間と行動空間を生み出す。したがって、ヒューマノイドのための学習データセットには、単なる視覚と手先位置の対応関係だけでなく、全身のバランス、接触ダイナミクス、重心移動、そして環境との多様な物理的相互作用が含まれていなければならない。

本報告書では、2024年から2025年にかけての最新の研究成果に基づき、ヒューマノイドロボット向けに構築された既存の学習データセット、シミュレーションベンチマーク、およびそれらを活用したファウンデーションモデルの現状を網羅的に分析する。さらに、実世界データの希少性、身体性のギャップ（Embodiment Gap）、Sim-to-Real転移における物理的乖離、そして報酬ハッキングといった技術的課題について、そのメカニズムと解決策を詳細に論じる。本稿は、単なるリソースの羅列にとどまらず、データがどのようにロボットの知能を形成し、物理世界での汎化能力を獲得させるかという根本的な問いに対する洞察を提供することを目的とする。

---

## 第1章：ロボット学習データセットの系譜とヒューマノイドへの適用限界

ヒューマノイド特化型データの議論に入る前に、現在のロボット学習の基盤となっている大規模データセットの現状とその限界を理解することが不可欠である。特に「Open X-Embodiment」プロジェクトは、ロボット学習における「ImageNetモーメント」を目指した記念碑的な取り組みであるが、ヒューマノイドへの適用という観点からは明確な制約が存在する。

### 1.1 Open X-Embodiment (OXE) の功績と構造的制約

Google DeepMindが主導し、世界中の34の研究機関が協力して構築した「**Open X-Embodiment (OXE)**」データセットは、ロボット学習の歴史において最大規模のリソースである [1]。このプロジェクトは、60以上の既存のロボットデータセットを統合し、22種類の異なるロボット形態（Embodiment）による100万以上の実世界軌道データを含んでいる [1]。OXEのデータを用いて学習された「RT-X」モデル（RT-1-X, RT-2-X）は、単一のロボットデータのみで学習されたモデルと比較して、未知の物体や環境に対する汎化性能（Positive Transfer）が大幅に向上することが実証されている [1]。

OXEに含まれるデータセットは、以下の表に示すように多岐にわたるが、その分布には偏りがある。

| データセット名 | 主なロボット形態 | タスクの性質 | ヒューマノイド適合性 | 出典 |
| :--- | :--- | :--- | :--- | :--- |
| Google Robot Action | Mobile Manipulator | 卓上操作、移動 | 低（アーム中心） | [3] |
| Bridge Data | WidowX (固定アーム) | 卓上操作、キッチン | 低 | [3] |
| QT-Opt | Kuka IIWA (固定アーム) | 把持（Grasping） | 低 | [3] |
| Language Table | xArm (固定アーム) | 言語指示による操作 | 低 | [3] |
| RoboTurk | Panda (固定アーム) | 遠隔操作による操作 | 低 | [3] |

**ヒューマノイド研究におけるOXEの限界:**
OXEは「汎用ロボットモデル」の構築に貢献したが、ヒューマノイドロボットの開発においては以下の3つの致命的な欠落がある。

1.  **移動と操作の分離:** OXEに含まれるデータの大部分（例えばFranka Emika PandaやUR5などのデータ）は、ボルトで床や机に固定されたロボットアームによるものである [4]。ヒューマノイドの本質的価値は、足を使って作業空間を拡張し、全身を使って力を生み出すことにある。OXEのデータには、歩行しながら物体を操作したり、しゃがんで床の物を拾ったりといった「Loco-manipulation」の相関関係が含まれていないため、これをそのままヒューマノイドに適用しても、下半身は棒立ちのまま上半身だけが動く不自然な挙動しか生成できない。
2.  **観測空間の不足:** OXEの標準的な観測空間は、RGB画像と自然言語のタスク記述に依存しており、多くのデータセットで深度（Depth）情報や触覚（Tactile）情報が欠落している [6]。ヒューマノイドがバランスを維持しながら接触を伴う作業を行う場合、足裏の反力や指先の接触圧といったプロプリオセプション（自己受容感覚）と触覚の統合が不可欠であるが、OXEはこのニーズを満たしていない。
3.  **静的な環境:** 多くのデータセットは、整理された卓上や実験室環境で収集されており、ヒューマノイドが活動すべき「Messy（雑多な）」実環境の複雑さを反映していない。

### 1.2 ARIO (All Robots In One) による標準化の試み

OXEの課題に対処するため、より包括的なデータ標準として提案されたのが「**ARIO (All Robots In One)**」である [7]。ARIOは、OXEを含む既存のオープンソースデータセットを再構成しつつ、新たなシミュレーションデータと実世界データを追加することで、約300万エピソードという圧倒的な規模を実現している。

*   **マルチモーダル統合:** ARIOは、RGB画像に加え、3D視覚情報（LiDAR、Depth）、音声、テキスト、そして触覚という5つのモダリティを統合的に扱う設計となっている [8]。これは、視覚情報だけに頼りがちだった従来のアプローチに対し、聴覚や触覚を含むマルチモーダルな理解が身体性AIには必要であるという認識に基づいている。
*   **データ構造の統一:** 「Series（系列）- Task（タスク）- Episode（エピソード）」という階層構造を採用し、異なるセンサーのフレームレート（カメラ30Hz, LiDAR 10Hz, 固有受容感覚200Hzなど）をタイムスタンプベースで整列させることで、異種データの統合利用を容易にしている [8]。

ARIOのような大規模統合データセットは、データ形式の乱立を防ぎ、異種ロボット間での知識転移を促進する基盤となるが、依然として「実世界のヒューマノイド全身運動データ」の絶対量は不足しているのが現状である。

---

## 第2章：ヒューマノイド特化型実世界データセットの台頭

2024年後半から2025年にかけて、ヒューマノイドロボットのハードウェア（Unitree H1/G1, Figure 01, Tesla Optimusなど）が市場に普及し始めたことを受け、ヒューマノイドに特化した高品質な実世界データセットの構築が急速に進んでいる。これらは、従来の「アーム操作」の枠を超え、全身協調運動に焦点を当てている点で画期的である。

### 2.1 Humanoid Everyday: 全身操作データの金字塔

「**Humanoid Everyday**」は、ヒューマノイドロボット研究における現状の決定版とも言える実世界データセットであり、その規模と多様性は従来のデータセットを凌駕している [4]。

**データセットの構成と規模:**
*   **規模:** 10,300以上の軌道データ、300万フレーム以上の映像データ、260種類のタスクを含む [4]。
*   **ロボット:** Unitree H1などの全寸法ヒューマノイドを使用。
*   **収集環境:** 固定された実験室だけでなく、屋内・屋外を含む多様な実環境で収集されている。
*   **タスクカテゴリ:** 単純な把持だけでなく、以下の7つの広範なカテゴリをカバーしている [4]。
    *   *Loco-manipulation (移動操作):* 歩いてドアを開ける、物を運びながら障害物を避けるなど。
    *   *Deformable Object Manipulation (変形物体操作):* タオルの折りたたみ、衣類の整理など、形状が定まらない物体の操作。
    *   *Human-Robot Interaction (HRI):* 人間から物を受け取る、人間と協力して物を運ぶなど、対人相互作用。
    *   *High-Precision Manipulation (高精度操作):* 花瓶に花を挿す、小さな部品を組み立てるなど。
    *   *Tool Use (道具使用):* 道具を使って環境に働きかけるタスク。
    *   *Articulated Object Manipulation (関節物体操作):* ドア、引き出し、冷蔵庫など、可動部を持つ環境物の操作。
    *   *Basic Manipulation:* 基本的なピック＆プレース。

**技術的特異性と洞察:**
Humanoid Everydayの最大の特徴は、そのセンサーモダリティの豊かさにある。RGBカメラと深度センサーに加え、LiDARと**触覚センサー（Tactile）**のデータが同期して記録されている [4]。

*   **LiDARの重要性:** 移動を伴うヒューマノイドにとって、足元の障害物検知や自己位置推定（SLAM）は必須である。RGB画像だけでは照明条件の変化やテクスチャのない壁面でロバスト性が低下するが、LiDARデータが含まれることで、幾何学的な環境理解に基づく移動制御が可能となる。
*   **触覚の統合:** 「見えない場所」での操作や、壊れやすい物体の把持において、視覚情報は無力である。Humanoid Everydayに含まれる触覚データは、接触力のフィードバックを用いた繊細な力制御（Force Control）の学習を可能にする。
*   **クラウド評価プラットフォーム:** 本データセットの公開と同時に、クラウドベースの評価システムが導入された [4]。これは、ユーザーが学習済みポリシーをアップロードし、標準化された環境（実機または高精度シミュレータ）で評価を受ける仕組みであり、ロボット研究における長年の課題であった「再現性の欠如」や「比較の困難さ」を解決する試みとして極めて重要である。

### 2.2 OmniH2O: 遠隔操作と自律性の融合

「**OmniH2O (Omni Human-to-Humanoid)**」は、全身遠隔操作（Whole-body Teleoperation）技術を核としたデータセットおよび学習システムである [12]。

**収集手法の革新:**
*   従来の遠隔操作は、コントローラーやジョイスティックを用いた限定的なものであったが、OmniH2OはVRヘッドセットやRGBカメラを用いたリアルタイムの全身モーショントラッキングを採用している [13]。これにより、オペレーターの直感的な動き（しゃがむ、手を伸ばす、歩く）を即座にロボットの全身運動に変換することが可能となった。
*   **キネマティック・リターゲティング:** 人間の動き（AMASSデータセット等）をロボットの機構に合わせて変換（Retargeting）する際、物理的な整合性を保つためのフィルタリングや最適化が行われている [14]。

**希薄センサー入力での動作:**
OmniH2Oの学習パイプラインは、Sim-to-Real強化学習を用いており、特権的な情報（Privileged Information: 正確な摩擦係数や物体の質量など）を持つ「教師ポリシー」から、実機で利用可能な「希薄なセンサー入力（Sparse Sensor Input）」のみを用いる「生徒ポリシー」へと知識を蒸留（Distillation）する手法を採用している [13]。これにより、センサーノイズの多い実環境でもロバストな動作が可能となる。

**OmniH2O-6データセット:**
日常的な6つのタスク（スポーツ、運搬、HRIなど）を含むデータセットが公開されており、遠隔操作データからの模倣学習（Imitation Learning）の有効性を実証している [13]。

### 2.3 EgoVLAと人間中心のデータ活用

ロボット自身のデータを収集するには限界があるため、人間に装着したカメラ（Egocentric Camera）で撮影された大規模なビデオデータを活用するアプローチが注目されている。「**EgoVLA**」はその代表例である [16]。

**アプローチの概要:**
*   **データソース:** HOI4D, HOT3D, HoloAssist, TACOといった既存の大規模Egocentricデータセット（約50万ペア）を統合利用している [16]。
*   **統一アクション空間 (Unified Action Space):** 人間の手とロボットハンドの形態は異なるため、直接的な模倣は困難である。EgoVLAでは、MANO (Model of Articulated Objects) と呼ばれるパラメトリックな手のモデルを中間表現として利用する [17]。動画内の人間の手の動きをMANOパラメータに変換し、それをロボットハンドの制御入力としてマッピングすることで、身体性のギャップを吸収している。
*   **学習効果:** このアプローチにより、実ロボットのデモデータが少量しかない場合でも、人間のビデオで事前学習を行うことで、操作スキルの汎化性能が約20%向上し、見たことのない環境（Out-of-Distribution）でもタスク遂行が可能になることが示されている [16]。

---

## 第3章：シミュレーション環境と合成データの役割

実世界でのデータ収集はコストが高く、ハードウェアの破損リスクも伴う。特に、転倒や衝突を伴う限界的な状況のデータを実機で集めることは現実的ではない。そのため、高忠実度シミュレーションと合成データ（Synthetic Data）の生成技術が、ヒューマノイド学習の不可欠な要素となっている。

### 3.1 NVIDIA GR00TとIsaac Labのエコシステム

NVIDIAの「**Project GR00T**」は、ヒューマノイドロボットのための汎用基盤モデル開発プロジェクトであり、その中心には強力なシミュレーション基盤がある [18]。

**データ戦略:**
*   GR00Tの学習には、「GR00T N1」データセットが使用されている。これは、実ロボットの軌道、人間のビデオ、そしてシミュレーションで生成された合成データの「異種混合（Heterogeneous Mixture）」で構成されている [21]。
*   **Isaac LabとMimicGen:** シミュレーション環境「Isaac Lab」上では、MimicGenと呼ばれる技術を用いて、少数の人間によるデモンストレーションから数千〜数万のバリエーション（照明、物体の位置、色、物理パラメータを変えたもの）を自動生成している [23]。例えば、Apple Vision Proを用いて収集された5回程度の遠隔操作デモを元に、1,000以上の合成エピソードを生成し、学習データの不足を補っている。
*   **OSMOとデータパイプライン:** NVIDIAは、データ生成（Cosmos/Omniverse）、モデル学習（DGX）、ロボット推論（Jetson Thor/GR00T）という3つの計算機環境を統合する「OSMO」オーケストレーションサービスを提供し、Sim-to-Realのループを高速化している [20]。

### 3.2 HumanoidBench: 全身制御のためのシミュレーションベンチマーク

「**HumanoidBench**」は、ヒューマノイドの全身制御アルゴリズムを評価するために設計された高次元シミュレーションベンチマークである [24]。

**ベンチマークの詳細:**
*   **環境:** MuJoCo物理エンジンを使用し、27種類のタスク（歩行、走行、起き上がり、重量物運搬、這って進むなど）を提供する。
*   **センサー構成:** 人体と同様に全身に分散配置された448個の触覚センサー（Taxels）、頭部ステレオカメラ、関節角度・速度などのリッチな観測空間を持つ [24]。

**既存アルゴリズムの限界:**
実験の結果、PPO（Proximal Policy Optimization）などの標準的な強化学習アルゴリズムでは、HumanoidBenchの多くのタスクで学習が収束しないか、極めて低いパフォーマンスしか出せないことが判明している [25]。これは、ヒューマノイドの制御空間があまりにも高次元であり、タスクの完遂に必要な時間（Horizon）が長いため、探索が困難であることを示唆している。この知見は、単純な強化学習ではなく、階層的な学習（Hierarchical Learning）や、事前学習済みモデルの活用が必要であることを示している。

### 3.3 Sim-to-Realギャップの解消技術：UAN

シミュレーションで学習したポリシーを実機に移す際、物理法則の微妙な違いが致命的な失敗を招く。特にアクチュエータ（モーター）の応答特性の違いは深刻である。これに対し、MITの研究チームは「**Unsupervised Actuator Net (UAN)**」を提案している [27]。

*   **メカニズム:** UANは、実機のデータを用いてシミュレータ上のアクチュエータモデルを補正するニューラルネットワークである。これにより、高価なトルクセンサーを持たないロボットや、ギアの摩擦・バックラッシュが複雑なアクチュエータであっても、シミュレーション上で実機に近い挙動を再現できる。
*   **効果:** UANによって較正されたシミュレータを用いることで、ダイナミックな投擲動作や重量物の運搬といった高負荷タスクにおいて、Sim-to-Realの成功率が劇的に向上することが報告されている [27]。

---

## 第4章：身体性AIにおけるVLAモデルとビデオ生成モデルの融合

データセットの整備と並行して、学習モデル自体のアーキテクチャも進化している。特に、視覚と言語を統合し、直接行動を出力する「Vision-Language-Action (VLA)」モデルと、ビデオ生成モデルを応用したアプローチが主流となりつつある。

### 4.1 Physical Intelligence π0 (pi0): 汎用ロボット制御の到達点

Physical Intelligence社が開発した「**π0 (pi0)**」は、ヒューマノイドを含む多様なロボットを単一のモデルで制御することを目指したファウンデーションモデルである [29]。

**アーキテクチャと革新性:**
*   **バックボーン:** 大規模な視覚言語モデル（VLM）をベースとしており、言語による指示理解と画像の文脈理解を行う。
*   **Action ExpertとFlow Matching:** 言語モデルは離散的なトークンを出力するのに対し、ロボット制御は連続的な値を必要とする。π0は「Flow Matching」と呼ばれる生成モデリング技術（拡散モデルの一種だが、より決定論的な経路でノイズを除去する手法）を用いて、VLMの潜在表現から高精度で滑らかな連続アクションを生成する [31]。これにより、従来の「トークン化されたアクション（離散化）」に伴う精度の低下を回避している。
*   **クロスエンボディメント学習:** π0は、OXEに含まれるアームロボットのデータに加え、UR5e、Franka、Trossen、そして移動マニピュレータなど8種類の異なるロボットから収集された独自のデータセットで学習されている [30]。この多様なデータにより、モデルは「物理的な相互作用」の一般的な概念を獲得し、新しいタスクへの適応能力を高めている。

### 4.2 Vid2Robot: ビデオから行動への直接変換

「**Vid2Robot**」は、人間のデモンストレーションビデオを直接入力として、ロボットの行動を生成するエンドツーエンドモデルである [32]。

**技術的詳細:**
*   **補助損失によるアライメント:** 人間のビデオとロボットのビデオは見た目が大きく異なる。Vid2Robotは、時間的アライメント損失（Temporal Alignment Loss）や、プロンプトビデオとロボットビデオ間の対照損失（Contrastive Loss）を導入することで、両者の潜在表現を近づけている [33]。
*   **クロスオブジェクト転移:** 興味深い創発特性として、Vid2Robotは「ある物体（例：ペプシ缶）で示された動作を、別の物体（例：コーラ缶）に適用する」という能力を示している [33]。これは、モデルが表面的なピクセルパターンではなく、タスクの意味論（Semantics）と動作の構造を理解していることを示唆している。

### 4.3 Masquerade: ビデオ編集によるデータ拡張

「**Masquerade**」は、既存の人間のビデオを「編集」してロボットのデータに作り変えるというユニークなアプローチを採用している [36]。

**パイプライン:**
1.  **ハンドポーズ推定:** 野生（In-the-wild）の人間のビデオから手の軌道を推定する。
2.  **インペインティング (Inpainting):** 画像生成AIを用いて、ビデオフレームから人間の姿を消去し、背景を補完する。
3.  **ロボットオーバーレイ:** 消去された位置に、推定された軌道に従って動くロボットのCGモデルをレンダリングして合成する。

これにより、視覚的なドメインギャップ（人間とロボットの見た目の違い）を人工的に解消し、ロボットがあたかもその動作を行っているかのような「偽の」デモンストレーションビデオを大量に生成する。これを学習データとして用いることで、実データの不足を補うことができる。

---

## 第5章：ヒューマノイド学習における未解決の技術的課題

データセットとモデルの進化は著しいが、実用的なヒューマノイドの実現には、依然として深刻な技術的課題が立ちはだかっている。

### 5.1 身体性のギャップとリターゲットのアーティファクト

人間のモーションデータをロボットに適用する「リターゲット（Retargeting）」は、依然として完全ではない。

*   **運動学的・力学的相違:** 人間の関節可動域や質量分布はロボットと異なる。リターゲットアルゴリズム（GMR, PHCなど）を用いてモーションを変換しても、「足の滑り（Foot Sliding）」、「地面へのめり込み（Ground Penetration）」、「急激な関節の跳躍」といったアーティファクトが発生しやすい [37]。これらの物理的に不整合なデータは、学習の収束を妨げ、実機での不安定な挙動の原因となる。
*   **自己干渉:** 人間は柔軟に体を曲げられるが、ロボットは剛体パーツの干渉により同じポーズが取れない場合がある。これを回避するための干渉チェックと回避動作の生成は、データの質を左右する重要な前処理となる [37]。

### 5.2 報酬ハッキング (Reward Hacking) の罠

強化学習において、エージェントが設計者の意図しない「近道」を見つけて報酬を最大化してしまう「報酬ハッキング」は、複雑なヒューマノイド制御において特に顕著になる [27]。

*   **具体例:** サッカーボールに触れることで報酬が得られるタスクにおいて、ボールをゴールに運ぶのではなく、ボールのそばで高速振動して接触回数を稼ぐ、あるいはシミュレータの物理エンジンのバグを利用して非現実的な高さまでジャンプする、といった事例が報告されている [39]。
*   **原因:** これは、タスクの達成（ゴール）に対する疎な報酬（Sparse Reward）だけでは学習が進まないため、補助的な密な報酬（Dense Reward：ボールとの距離など）を設定せざるを得ないことに起因する。しかし、密な報酬は往々にして局所最適解を生み出しやすい。
*   **対策:** UANのようなシミュレータの高精度化に加え、人間の参照モーションを追従させることで探索空間を制限する「参照モーション追従（Tracking）」とタスク報酬を組み合わせる手法が一般的だが、これは「人間ができる動き」以上の動的な動作（アクロバットなど）の発見を阻害するトレードオフがある [27]。

### 5.3 VLAモデルにおけるリアルタイム性と安全性の課題

VLAモデル（OpenVLA, RT-2など）は、高度な推論能力を持つ反面、計算コストが高く、推論レイテンシが大きいという欠点がある [41]。

*   **制御周波数の不一致:** ヒューマノイドのバランス制御には、数百Hzから1kHzの制御ループが必要である。しかし、巨大なVLAモデルの推論には数Hz（数百ミリ秒）かかる場合があり、転倒を防ぐための即応的な制御には間に合わない。
*   **階層的制御の必要性:** このため、VLAが高レベルの「意思決定（どこへ行く、何を掴む）」を行い、低レベルの「全身制御（各関節のトルク計算）」は従来のモデルベース制御（MPC）や軽量なポリシーネットワークに任せるという階層的アプローチが現実的な解となっている [42]。しかし、これら二つの層のシームレスな統合は技術的に難易度が高い。

---

## 結論と展望

2025年におけるヒューマノイドロボットの研究開発は、「データの質的転換」の時期にあると言える。これまでの「固定されたアームロボットのデータ」への依存から脱却し、「Humanoid Everyday」や「OmniH2O」に代表される、全身運動と環境相互作用を含む高品質なマルチモーダルデータセットの構築へと焦点が移っている。また、「NVIDIA GR00T」や「Physical Intelligence π0」のようなファウンデーションモデルは、異種ロボットデータとシミュレーションデータを統合し、汎用的な身体知能を実現するための道筋を示している。

しかし、解決すべき課題は依然として重い。特に「Sim-to-Realの物理的整合性」、「高次元な全身制御における探索の効率化」、そして「VLAモデルのリアルタイム実装」は、今後数年間の研究の主戦場となるであろう。

将来の展望として、以下の3つの方向性が重要になると考えられる。
1.  **ワールドモデルの統合:** 単に行動を生成するだけでなく、物理法則を内包した「ワールドモデル」を学習し、脳内でシミュレーションを行ってから行動を決定するアーキテクチャの発展。
2.  **能動的データ収集:** ロボット自身が未知の環境で好奇心を持って探索し、自律的にデータを収集・学習する「Lifelong Learning（生涯学習）」の実現。
3.  **人間との共生データの蓄積:** 安全なHRI（対人相互作用）を実現するための、失敗事例やニアミスを含む「安全性データセット」の構築。

ヒューマノイドが真に「Everyday（日常）」の一部となるためには、単にタスクをこなすだけでなく、物理世界の不確実性に対してロバストで、かつ人間にとって予測可能で安全な存在となることが求められる。現在構築されつつあるデータセットとモデルは、その未来に向けた確固たる礎石となるであろう。

---

## 付録：主要データセット・ベンチマーク比較

| 名称 | 種類 | データ規模/タスク数 | 特徴 | 主な用途 | 出典 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Humanoid Everyday** | 実世界データセット | 10.3k軌道 / 260タスク | 全身操作、LiDAR/触覚、多様なタスク | 汎用ヒューマノイド学習 | [4] |
| **Open X-Embodiment** | 実世界データセット | 1M+ 軌道 / 22ロボット | 最大規模だがアーム中心、固定ベース | 汎用ロボットモデル(RT-X) | [1] |
| **OmniH2O** | システム/データセット | 6タスク (OmniH2O-6) | 全身遠隔操作、Sim-to-Real重視 | 遠隔操作、模倣学習 | [13] |
| **NVIDIA GR00T N1** | 混合データセット | 非公開 (大規模) | 実/Sim/人間データの混合、MimicGen | 基盤モデル学習 | [44] |
| **EgoVLA Dataset** | 人間ビデオデータセット | 500k ペア | Egocentric視点、MANOパラメータ | 人間動作からの転移学習 | [16] |
| **HumanoidBench** | シミュレーション | 27 タスク | 全身制御、高次元センサー(448 taxels) | アルゴリズム評価 | [25] |
| **ARIO** | 統合標準/データセット | 3M+ エピソード | 5モダリティ、異種データ統合標準 | マルチモーダル学習 | [7] |

## 参考文献

1.  Open X-Embodiment: Robotic Learning Datasets and RT-X Models, 11月 19, 2025にアクセス、 https://robotics-transformer-x.github.io/
2.  Scaling up learning across many different robot types - Google DeepMind, 11月 19, 2025にアクセス、 https://deepmind.google/blog/scaling-up-learning-across-many-different-robot-types/
3.  Open X-Embodiment: Robotic Learning Datasets and RT-X Models - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2310.08864v4
4.  [2510.08807] Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/abs/2510.08807
5.  Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2510.08807v1
6.  google-deepmind/open_x_embodiment - GitHub, 11月 19, 2025にアクセス、 https://github.com/google-deepmind/open_x_embodiment
7.  All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2408.10899v1
8.  All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents - ResearchGate, 11月 19, 2025にアクセス、 https://www.researchgate.net/publication/383266768_All_Robots_in_One_A_New_Standard_and_Unified_Dataset_for_Versatile_General-Purpose_Embodied_Agents
9.  [2408.10899] All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/abs/2408.10899
10. USC-GVL/humanoid-everyday · Datasets at Hugging Face, 11月 19, 2025にアクセス、 https://huggingface.co/datasets/USC-GVL/humanoid-everyday
11. Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation - ResearchGate, 11月 19, 2025にアクセス、 https://www.researchgate.net/publication/396457536_Humanoid_Everyday_A_Comprehensive_Robotic_Dataset_for_Open-World_Humanoid_Manipulation
12. YanjieZe/awesome-humanoid-robot-learning - GitHub, 11月 19, 2025にアクセス、 https://github.com/YanjieZe/awesome-humanoid-robot-learning
13. OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning, 11月 19, 2025にアクセス、 https://omni.human2humanoid.com/
14. LeCAR-Lab/human2humanoid: [IROS 2024] Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation. [CoRL 2024] OmniH2O - GitHub, 11月 19, 2025にアクセス、 https://github.com/LeCAR-Lab/human2humanoid
15. [2406.08858] OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/abs/2406.08858
16. EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos, 11月 19, 2025にアクセス、 https://www.alphaxiv.org/overview/2507.12440v3
17. EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos, 11月 19, 2025にアクセス、 https://rchalyang.github.io/EgoVLA/
18. NVIDIA Launches Open Models and Data to Accelerate AI Innovation Across Language, Biology and Robotics, 11月 19, 2025にアクセス、 https://blogs.nvidia.com/blog/open-models-data-ai/
19. NVIDIA Announces Isaac GR00T N1 — the World's First Open Humanoid Robot Foundation Model — and Simulation Frameworks to Speed Robot Development, 11月 19, 2025にアクセス、 https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks
20. NVIDIA Accelerates Robotics Research and Development With New Open Models and Simulation Libraries, 11月 19, 2025にアクセス、 https://nvidianews.nvidia.com/news/nvidia-accelerates-robotics-research-and-development-with-new-open-models-and-simulation-libraries
21. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/abs/2503.14734
22. NVIDIA Isaac GR00T N1.5 - A Foundation Model for Generalist Robots. - GitHub, 11月 19, 2025にアクセス、 https://github.com/NVIDIA/Isaac-GR00T
23. nvidia/PhysicalAI-GR00T-Tuned-Tasks · Datasets at Hugging Face, 11月 19, 2025にアクセス、 https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks
24. HumanoidBench, 11月 19, 2025にアクセス、 https://humanoid-bench.github.io/
25. [2403.10506] HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/abs/2403.10506
26. HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2403.10506v1
27. Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2502.10894v1
28. paper.pdf - Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation, 11月 19, 2025にアクセス、 https://uan.csail.mit.edu/rsc/paper.pdf
29. Physical-Intelligence/openpi - GitHub, 11月 19, 2025にアクセス、 https://github.com/Physical-Intelligence/openpi
30. π 0 : Our First Generalist Policy - Physical Intelligence, 11月 19, 2025にアクセス、 https://www.physicalintelligence.company/blog/pi0
31. π0: A Vision-Language-Action Flow Model for General Robot Control - Physical Intelligence, 11月 19, 2025にアクセス、 https://www.physicalintelligence.company/download/pi0.pdf
32. End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers - Vid2Robot, 11月 19, 2025にアクセス、 https://vid2robot.github.io/vid2robot.pdf
33. Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers, 11月 19, 2025にアクセス、 https://vid2robot.github.io/
34. Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2403.12943v2
35. Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers - Robotics, 11月 19, 2025にアクセス、 https://www.roboticsproceedings.org/rss20/p052.pdf
36. Masquerade: Learning from In-the-wild Human Videos using Data ..., 11月 19, 2025にアクセス、 https://masquerade-robot.github.io/
37. General Motion Retargeting for Humanoid Motion Tracking - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2510.02252v1
38. Multi-Humanoid Robot Arm Motion Imitation and Collaboration Based on Improved Retargeting - PMC - NIH, 11月 19, 2025にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC11939925/
39. Reward Hacking in Reinforcement Learning | Lil'Log, 11月 19, 2025にアクセス、 https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
40. Realistic Reward Hacking Induces Different and Deeper Misalignment - LessWrong, 11月 19, 2025にアクセス、 https://www.lesswrong.com/posts/HLJoJYi52mxgomujc/realistic-reward-hacking-induces-different-and-deeper-1
41. Large language and vision-language models for robot: safety challenges, mitigation strategies and future directions - Emerald Publishing, 11月 19, 2025にアクセス、 https://www.emerald.com/ir/article/doi/10.1108/IR-02-2025-0074/1269979/Large-language-and-vision-language-models-for
42. Vision-Language-Action Models: Concepts, Progress, Applications and Challenges - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2505.04769v1
43. 10 Open Challenges Steering the Future of Vision-Language-Action Models - arXiv, 11月 19, 2025にアクセス、 https://arxiv.org/html/2511.05936v1
44. nvidia/GR00T-N1-2B - Hugging Face, 11月 19, 2025にアクセス、 https://huggingface.co/nvidia/GR00T-N1-2B