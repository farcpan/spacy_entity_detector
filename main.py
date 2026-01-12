import pprint
import spacy
import time

text = """
    AWS公式の立場
    Amazon Bedrock は「顧客の入力（prompts）や出力（completions）をモデルの訓練に使わない」
    「モデルプロバイダーと顧客データを共有しない」と明言しています。
    さらに通信・保管ともに暗号化やVPC/PrivateLinkなどの隔離機構を提供しています。
    しかし残る実務的リスク：完全ゼロではありません。
    典型的なリスクは
    （1）アプリ実装/設定ミスによる漏洩、
    （2）評価や人手レビュー機能を用いた場合の人的露出、
    （3）一時的ストレージやログの扱い、
    （4）モデル出力（誤情報や個人情報のリーク）に伴うビジネスリスク、
    （5）サードパーティモデル固有のセキュリティ脆弱性等です。
    これらは設計・運用・契約で十分抑止・検出できます。

    お世話になっております。株式会社HOGEHOGEの鈴木です。
"""

"""
抽出結果: 
Amazon Bedrock Person
prompts Name_Other
PrivateLink Music
（1） Ordinal_Number
（2） Ordinal_Number
（3） Ordinal_Number
（4） Ordinal_Number
（5） Ordinal_Number

処理時間: 1.49 sec

※結果はいまいち。
"""

start = time.time()

## Model ---------------------------------------------
#nlp = spacy.load("ja_ginza")
nlp = spacy.load("ja_core_news_md")

## Checking Entity Label -----------------------------
#pprint.pprint(nlp.get_pipe('ner').labels)

## Inference -----------------------------------------
doc = nlp(text)

## Results -------------------------------------------
print("=" * 30)
for ent in doc.ents:
    if ent.label_ in ["PERSON", "ORG", "EVENT", "PRODUCT", "FAC"]:
        print("-" * 30)
        print(f"[{ent.label_}] {ent.text}")

print("=" * 30)
print(f"Elapsed time: {time.time() - start} [sec]")
