"""
POI推荐理由生成Agent
参考小红书本地生活地图场景 - "输入筛选 → 亮点提炼 → 评测迭代"工作流
"""

import os
import json
import anthropic

MODEL = "claude-sonnet-4-20250514"
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ─────────────────────────────────────────────
# Agent 1: 筛选Agent
# ─────────────────────────────────────────────

def filter_agent(reviews: list[str]) -> list[str]:
    """
    过滤低质量评论：
    - 硬规则：字数 < 10 直接剔除
    - 软规则：调用Claude判断是否内容泛化
    """
    # 硬规则：长度过滤
    length_passed = [r for r in reviews if len(r) >= 10]

    if not length_passed:
        return []

    # 软规则：Claude判断泛化内容
    reviews_json = json.dumps(length_passed, ensure_ascii=False)
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""你是评论质量筛选器。判断以下每条评论是否属于"内容泛化"。

内容泛化定义：语义模糊、无具体信息，如"很好"、"不错"、"还行"、"挺好的"、"值得推荐"、"下次还来"等。

评论列表（JSON数组）：
{reviews_json}

请逐条判断，返回一个JSON数组，每个元素格式为：
{{"review": "原文", "is_generic": true或false}}

只返回JSON，不要其他内容。"""
        }]
    )

    raw = response.content[0].text.strip()
    # 去掉可能的markdown代码块
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    judgements = json.loads(raw.strip())

    filtered = [item["review"] for item in judgements if not item["is_generic"]]
    return filtered


# ─────────────────────────────────────────────
# Agent 2: 生成Agent
# ─────────────────────────────────────────────

def generation_agent(poi_name: str, poi_category: str, filtered_reviews: list[str]) -> str:
    """
    基于筛选后评论，生成个性化推荐理由（≤30字）。
    若无有效评论，走兜底生成逻辑。
    """
    if filtered_reviews:
        reviews_text = "\n".join(f"- {r}" for r in filtered_reviews)
        prompt = f"""你是本地生活推荐文案专家。根据以下真实用户评论，为POI生成一条推荐理由。

POI名称：{poi_name}
POI类别：{poi_category}
用户评论：
{reviews_text}

要求：
1. 提炼评论中最有价值的具体亮点（菜品、体验、特色等）
2. 语言自然、有吸引力，适合推荐场景
3. 不超过30个汉字，不要引号
4. 只输出推荐理由文本，不要任何解释"""
    else:
        # 兜底逻辑：无有效评论时基于类别生成
        prompt = f"""你是本地生活推荐文案专家。为以下POI生成一条通用推荐理由。

POI名称：{poi_name}
POI类别：{poi_category}

要求：
1. 语言自然、有吸引力
2. 不超过30个汉字，不要引号
3. 只输出推荐理由文本，不要任何解释"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}]
    )

    recommendation = response.content[0].text.strip().strip('"').strip("\"")
    # 截断保险：超长时截到30字
    if len(recommendation) > 30:
        recommendation = recommendation[:30]
    return recommendation


# ─────────────────────────────────────────────
# Agent 3: 评测Agent
# ─────────────────────────────────────────────

def evaluation_agent(recommendation: str, poi_name: str, filtered_reviews: list[str]) -> dict:
    """
    对推荐理由按三个维度打分（0-3分），总分≤5标记为bad case。

    三维度：
    - 基础可用性：是否通顺、无幻觉、非空
    - 内容质量：是否有具体亮点，避免"环境很好"式泛化
    - 语言风格：是否自然、有感染力、符合推荐场景
    """
    reviews_text = "\n".join(f"- {r}" for r in filtered_reviews) if filtered_reviews else "（无有效评论）"

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""你是推荐文案质量评测专家。请对以下推荐理由进行评分。

POI名称：{poi_name}
参考评论：
{reviews_text}

待评测推荐理由："{recommendation}"

评分标准（每项0-3分）：

【基础可用性】
3分：语句通顺、无幻觉、内容与POI相关
2分：基本可读，有小瑕疵
1分：存在语病或轻微幻觉
0分：空内容、严重错误或完全不相关

【内容质量】
3分：有具体亮点（菜品名/特色/体验细节），信息量高
2分：有一定具体性，但不够鲜明
1分：较为泛化，缺少独特信息
0分：完全泛化如"环境很好""服务不错"

【语言风格】
3分：自然流畅、有感染力、符合本地生活推荐语气
2分：基本合适，但略显平淡
1分：生硬或过于正式
0分：完全不符合推荐场景

请返回JSON格式：
{{"基础可用性": 整数, "内容质量": 整数, "语言风格": 整数, "评分理由": "一句话"}}

只返回JSON，不要其他内容。"""
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())

    scores = {
        "基础可用性": int(result["基础可用性"]),
        "内容质量": int(result["内容质量"]),
        "语言风格": int(result["语言风格"]),
    }
    scores["total"] = sum(scores.values())
    return scores


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run_poi_agent(poi_name: str, poi_category: str, reviews: list[str]) -> dict:
    """
    完整的POI推荐理由生成流程。

    返回：
    {
        "recommendation": str,       # 推荐理由
        "scores": {                  # 三维度评分
            "基础可用性": int,
            "内容质量": int,
            "语言风格": int,
            "total": int
        },
        "bad_case": bool,            # 总分≤5则为bad case
        "filtered_reviews": list     # 筛选后的有效评论
    }
    """
    print(f"\n{'='*50}")
    print(f"POI: {poi_name}（{poi_category}）")
    print(f"{'='*50}")

    # Step 1: 筛选
    print(f"\n[筛选Agent] 输入 {len(reviews)} 条评论...")
    filtered = filter_agent(reviews)
    print(f"  → 筛选后保留 {len(filtered)} 条：")
    for r in filtered:
        print(f"    ✓ {r}")
    removed = [r for r in reviews if r not in filtered]
    for r in removed:
        print(f"    ✗ {r}（已过滤）")

    # Step 2: 生成
    print(f"\n[生成Agent] 基于有效评论生成推荐理由...")
    recommendation = generation_agent(poi_name, poi_category, filtered)
    print(f"  → 推荐理由：{recommendation}（{len(recommendation)}字）")

    # Step 3: 评测
    print(f"\n[评测Agent] 对推荐理由打分...")
    scores = evaluation_agent(recommendation, poi_name, filtered)
    bad_case = scores["total"] <= 5

    print(f"  → 基础可用性：{scores['基础可用性']}/3")
    print(f"  → 内容质量：{scores['内容质量']}/3")
    print(f"  → 语言风格：{scores['语言风格']}/3")
    print(f"  → 总分：{scores['total']}/9  {'⚠️  BAD CASE' if bad_case else '✅ 质量达标'}")

    return {
        "recommendation": recommendation,
        "scores": scores,
        "bad_case": bad_case,
        "filtered_reviews": filtered,
    }


def main():
    # Demo：一家火锅店，5条评论（混合高质量与泛化评论）
    poi_name = "蜀大侠火锅（三里屯店）"
    poi_category = "火锅"
    reviews = [
        "很好",                                                    # 泛化，应被过滤
        "不错",                                                    # 泛化，应被过滤
        "毛肚和鸭肠都很脆，蘸料的芝麻香特别突出，锅底是正宗成都老火锅味",  # 高质量
        "服务挺好的，下次还来",                                      # 泛化，应被过滤
        "自助蘸料台品类多，花生碎+香菜+蒜泥的组合是必试搭配，排队等位约30分钟",  # 高质量
    ]

    result = run_poi_agent(poi_name, poi_category, reviews)

    print(f"\n{'='*50}")
    print("最终输出结果：")
    print(f"{'='*50}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
