import os
import pandas as pd
from openai import OpenAI
import difflib

ratings = pd.read_csv("../dataset/Multimodal_Datasets/M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("../dataset/Multimodal_Datasets/M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]

client = OpenAI(
    api_key="sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""


def get_user_history_titles(user_id):
    df = ratings[(ratings.user_id == user_id)].sort_values("timestamp")
    watched_ids = df["movie_id"].tolist()
    title_map = movies.set_index("movie_id")["movie_title"].to_dict()
    return [title_map.get(mid, str(mid)) for mid in watched_ids]


def recommend_by_llm(user_id):
    history_titles = get_user_history_titles(user_id)
    if not history_titles:
        return []
    prompt = PROMPT_TEMPLATE.format("\n".join(history_titles))
    resp = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "你是专业的电影推荐系统。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    titles = [l.lstrip("- ") for l in lines][:10]
    return titles

def recommend_for_user(user_id, top_k=10):
    titles = recommend_by_llm(user_id)
    title_map = movies.set_index("movie_title")["movie_id"].to_dict()
    return [(title_map.get(t, t), t) for t in titles[:top_k]]

if __name__ == "__main__":
    user_id = 196
    recs = recommend_for_user(user_id, top_k=10)
    print(f"用户 {user_id} 推荐结果：")
    for i, (mid, title) in enumerate(recs, start=1):
        print(f"{i}. {mid} - {title}")
