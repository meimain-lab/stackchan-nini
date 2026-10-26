[English](README.md) | [中文](README.zh-CN.md) | 日本語

> [!IMPORTANT]
> **非商用利用に限ります。** ファームウェアには、非商用利用のみ許諾されている Emotion Ball の目のリングデータが含まれています。個人・教育などの非商用目的での利用・改変・共有は可能ですが、本プロジェクトやそれを搭載したハードウェアを販売することはできません。詳しくは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#emotion-ball-eye-ring-data-aora-bot-emotion-ball-community-license) を参照してください。

![Groki Bot](docs/images/banner.png)
<!-- demo-video
     Demo video to be added. After publishing, place the embed code (iframe or video) below this comment block.
-->

# Groki Bot

Groki Bot は、Grok Bot アプリにつないで直接会話できるオープンソースのデスクトップロボット（M5Stack CoreS3 + Stack-chan ベース）です。ファームウェア、PC 側ゲートウェイ、Grok Bot 連携が 1 つのリポジトリに入っています。

> 目の表現には [aora-bot](https://github.com/sam70361/aora-bot) Emotion Ball の

## 機能

- ネイティブ音声（macOS のみ）：Grok Bot アプリのエージェントと直接音声で会話できます。Mac 上のアプリのログイン状態を使うため、別の xAI 音声 API キーは不要です。
- ハンズフリー音声起動：「Hi Grok」と呼びかけるだけでいつでも対話を開始でき、PC 側ゲートウェイの軽量な音声検出アルゴリズムが起動ワードを検知します。
- 動的フェイストラッキング：ロボットの視線を自分に向けることができ、Mac のカメラで顔を認識して 2 軸サーボベースをリアルタイムに駆動します。
- 豊かな表情フィードバック：本体の表情エンジンが 15 種類の表情と瞬きを描画します。
- ブラウザからの簡単書き込みと OTA：ローカルのクロスコンパイル環境構築が不要で、Chrome や Edge から直接 Web でファームウェアを書き込め、Wi-Fi 経由のデバイス OTA 更新にも対応します。

## クイックスタート：Grok Bot ネイティブ音声で対話する

1. ファームウェアの書き込みと Wi-Fi 設定：デスクトップ版 Chrome または Edge で Web 書き込みページを開いてファームウェアを書き込み、設定画面で Wi-Fi に接続します。
2. Mac でゲートウェイを準備：Grok Bot デスクトップアプリにログイン済みの Mac で本リポジトリをクローンし、`gateway/` ディレクトリで依存関係を同期します。
   ```sh
   cd gateway && uv sync --all-extras
   ```
3. Grok 音声に切り替える：既定のバックエンドは Gemini なので、`gateway/.env` に `STACKCHAN_VOICE_BACKEND=gbot` とエージェント名と ID を設定します。Grok Bot アプリのログインを維持し、別ターミナルでログインヘルパーを常駐させ、ゲートウェイを起動して、BLE 設定ページでロボットの接続先を PC のゲートウェイに設定します。
4. 「Hi Grok」と話しかけて対話開始：ロボットに「Hi Grok」と話しかければすぐに会話を開始できます。環境変数の詳細設定、ログインブリッジの起動コマンド、元のバックエンドへの切り替え手順は[Grok Bot ネイティブ音声の設定](docs/quick-start.ja.md#grok-bot-ネイティブ音声の設定)を参照してください。

## ドキュメント

- [クイックスタートと設定](docs/quick-start.ja.md)
- [他の音声バックエンド](docs/voice-backends.ja.md)
- [機能](docs/features.ja.md)
- [ハードウェア](docs/hardware.ja.md)
- [ソースからのビルド](docs/build.ja.md)
- [よくある質問とトラブルシューティング](docs/troubleshooting.ja.md)
- [アーキテクチャ](docs/architecture.md)
- [リポジトリ構成](docs/repository.ja.md)
- [ゲートウェイ](gateway/README.md)

## ライセンス

[BSL-1.0](LICENSE) | [Gateway (MIT)](gateway/LICENSE) | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [docs/licenses.html](docs/licenses.html) | [repository](docs/repository.ja.md)

第三者のライセンスと帰属表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
