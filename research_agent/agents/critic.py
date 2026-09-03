import json
import re
from textwrap import dedent


class CriticAgent:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _parse_json(text):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def review(self, question, draft, evidence, analysis_dimensions=None):
        analysis_dimensions = analysis_dimensions or []
        evidence_summary = [{
            "id": f"E{i}",
            "source_type": item.get("source_type", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "score": item.get("overall_score", 0),
            "authority": item.get("authority", 0),
            "content": item.get("content", "")[:1600]
        } for i, item in enumerate(evidence[:12], 1)]

        prompt = dedent("""\
            你是Critic Agent。

            你的目标不是帮助Researcher证明原结论，而是主动寻找原结论的漏洞、证据缺口和过度推断。

            用户问题：
            {question}

            Researcher结果：
            {draft}

            用户要求覆盖的分析维度：
            {analysis_dimensions}

            当前证据：
            {evidence}

            请重点审查：
            1. 每个关键claim是否真的有对应证据；
            2. 引用编号是否支持该claim；
            3. 是否把相关性错误当作因果关系；
            4. 是否存在证据不足却表述过于确定的问题；
            5. 是否缺少重要反例或反方观点；
            6. 主要比较对象是否都有高质量证据；
            7. 开源项目是否缺少官方GitHub证据；
            8. 学术判断是否缺少论文证据；
            9. 是否存在明显过时信息或维护状态变化；
            10. 是否需要进一步检索才能得出可靠结论。
            11. 证据编号统一采用E1、E2、E3格式。
            12. 不要自行创造不存在的证据编号。
            13. 逐Claim检查其范围是否过宽、是否跨越多个分析维度。
            14. 逐个核对evidence_ids是否直接支持该Claim，不能把主题相关当作直接支持。
            15. valid_evidence_ids只能从该Claim原始evidence_ids中选择，不得新增编号。
            16. supported表示主张及限定条件被直接支持；partial表示只能支持更有限的表述；unsupported不得进入Verified Claims。

            Researcher的每个claim都有claim_id，例如C1、C2。

            如果某个claim缺乏直接证据支持，
            unsupported_claims必须使用原始claim_id，
            不能自己重新编号。

            例如：

            "unsupported_claims": [
            {{
                "claim_id": "C2",
                "claim": "...",
                "reason": "证据不足"
            }}
            ]

            只输出JSON：
            {{
                "sufficient": false,
                "critique": "总体审查",
                "claim_reviews": [
                    {{
                        "claim_id": "C1",
                        "verdict": "supported",
                        "valid_evidence_ids": ["E1", "E3"],
                        "reason": "这些证据直接支持该原子主张"
                    }},
                    {{
                        "claim_id": "C2",
                        "verdict": "partial",
                        "valid_evidence_ids": ["E4"],
                        "reason": "证据只支持更有限的结论"
                    }}
                ],
                "unsupported_claims": [
                    {{
                    "claim_id": "C2",
                    "claim": "具体claim",
                    "reason": "为什么证据不足"
                    }}
                ],
                "missing_perspectives": [],
                "followup_queries": [],
                "followup_github_repos": [],
                "followup_github_queries": [],
                "followup_paper_queries": []
            }}

            如果需要补充证据，将sufficient设为false，并生成针对性的补充检索请求。
        """).format(
            question=question,
            draft=json.dumps(draft, ensure_ascii=False),
            analysis_dimensions=json.dumps(
                analysis_dimensions,
                ensure_ascii=False
            ),
            evidence=json.dumps(evidence_summary, ensure_ascii=False)
        )

        result = self._parse_json(
            self.llm.generate(prompt, max_new_tokens=1800)
        )

        if result is None:
            claims = draft.get("claims", []) if isinstance(draft, dict) else []
            return {
                "sufficient": False,
                "critique": "Critic输出解析失败，无法完成逐Claim证据核验。",
                "claim_reviews": [],
                "unsupported_claims": [
                    {
                        "claim_id": item.get("claim_id", ""),
                        "claim": item.get("claim", ""),
                        "reason": "Critic未能完成逐Claim核验"
                    }
                    for item in claims
                    if isinstance(item, dict)
                ],
                "missing_perspectives": ["逐Claim证据核验未完成"],
                "followup_queries": [],
                "followup_github_repos": [],
                "followup_github_queries": [],
                "followup_paper_queries": []
            }

        claims = draft.get("claims", []) if isinstance(draft, dict) else []
        claims_by_id = {
            str(item.get("claim_id", "")).upper(): item
            for item in claims
            if isinstance(item, dict) and item.get("claim_id")
        }
        raw_reviews = result.get("claim_reviews", [])
        if not isinstance(raw_reviews, list):
            raw_reviews = []

        reviews_by_id = {}
        evidence_count = len(evidence)
        for review in raw_reviews:
            if not isinstance(review, dict):
                continue
            claim_id = str(review.get("claim_id") or "").upper()
            claim = claims_by_id.get(claim_id)
            if not claim or claim_id in reviews_by_id:
                continue

            researcher_ids = {
                str(item).strip().upper()
                for item in claim.get("evidence_ids", [])
            }
            valid_ids = []
            raw_valid_ids = review.get("valid_evidence_ids", [])
            if not isinstance(raw_valid_ids, list):
                raw_valid_ids = [raw_valid_ids]
            for evidence_id in raw_valid_ids:
                normalized_id = str(evidence_id).strip().upper()
                match = re.fullmatch(r"E([1-9]\d*)", normalized_id)
                if (
                    match
                    and int(match.group(1)) <= evidence_count
                    and normalized_id in researcher_ids
                    and normalized_id not in valid_ids
                ):
                    valid_ids.append(normalized_id)

            verdict = str(review.get("verdict") or "unsupported").lower()
            if verdict not in {"supported", "partial", "unsupported"}:
                verdict = "unsupported"
            if verdict in {"supported", "partial"} and not valid_ids:
                verdict = "unsupported"

            reviews_by_id[claim_id] = {
                "claim_id": claim_id,
                "verdict": verdict,
                "valid_evidence_ids": valid_ids[:5],
                "reason": str(review.get("reason") or "").strip()
            }

        for claim_id in claims_by_id:
            if claim_id not in reviews_by_id:
                reviews_by_id[claim_id] = {
                    "claim_id": claim_id,
                    "verdict": "unsupported",
                    "valid_evidence_ids": [],
                    "reason": "Critic未返回该Claim的逐项证据审查"
                }

        claim_reviews = list(reviews_by_id.values())
        unsupported = result.get("unsupported_claims", [])
        if not isinstance(unsupported, list):
            unsupported = []
        unsupported_ids = {
            str(item.get("claim_id", "")).upper()
            for item in unsupported
            if isinstance(item, dict)
        }
        for review in claim_reviews:
            if review["verdict"] != "unsupported":
                continue
            claim_id = review["claim_id"]
            if claim_id in unsupported_ids:
                continue
            claim = claims_by_id[claim_id]
            unsupported.append({
                "claim_id": claim_id,
                "claim": claim.get("claim", ""),
                "reason": review["reason"] or "缺少可确认的直接支持证据"
            })

        result["claim_reviews"] = claim_reviews
        result["unsupported_claims"] = unsupported
        return result
