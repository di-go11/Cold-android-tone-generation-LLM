import requests

def search_narou_by_keyword(keyword: str, limit: int = 10):
    url = "https://api.syosetu.com/novelapi/api/"
    params = {
        "out": "json",
        "keyword": keyword,
        "lim": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # 最初の要素はヘッダー情報なのでスキップ
    for novel in data[1:]:
        print(f"📘 タイトル: {novel['title']}")
        print(f"✍️ 作者: {novel['writer']}")
        print(f"📝 あらすじ: {novel['story'][:100]}...")
        print(f"🔑 キーワード:{novel['keyword']}")
        print(f"🔗 URL: https://ncode.syosetu.com/{novel['ncode'].lower()}/")
        print("-" * 40)

# 使用例
search_narou_by_keyword("無表情メイド", limit=10)
