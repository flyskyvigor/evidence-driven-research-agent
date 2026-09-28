import json
import re


class ResearcherAgent:
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

    @staticmethod
    def _build_evidence_text(evidence, content_limit=3200):
        parts = []

        for i, item in enumerate(evidence[:12], 1):
            parts.append(
                f"[E{i}]\n"
                f"Score: {item.get('overall_score', 0):.3f}\n"
                f"Type: {item.get('source_type', '')}\n"
                f"Source: {item.get('source_name', '')}\n"
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Content: {item.get('content', '')[:content_limit]}"
            )

        return "\n\n".join(parts)

    def research(
        self,
        question,
        plan,
        evidence,
        analysis_dimensions=None,
        previous_critique=""
    ):
        evidence_text = self._build_evidence_text(evidence)
        analysis_dimensions = analysis_dimensions or plan.get(
            "analysis_dimensions",
            []
        )

        prompt = f"""你是Researcher Agent，负责基于可靠证据形成研究结论。

用户问题：
{question}

研究计划：
{json.dumps(plan, ensure_ascii=False)}

必须覆盖的分析维度：
{json.dumps(analysis_dimensions, ensure_ascii=False)}

上一轮Critic反馈：
{previous_critique or "无"}

当前证据：
{evidence_text}

任务：
1. 只能根据当前提供的证据回答，不允许补充或虚构不存在的事实。
2. 明确区分事实、推断和暂时判断。
3. 优先使用高分、一手来源。
4. 如果证据存在冲突，应明确说明。
5. 上一轮Critic指出的问题必须认真处理。
6. 证据编号只能使用E1、E2等当前真实存在的编号。
7. Claim必须原子化：一个Claim只表达一个可由证据核验的判断，不能跨越多个分析维度。
8. 复杂问题通常形成3到6个Claim并尽量覆盖全部维度，但不得为凑数量编造结论。
9. 每个Claim必须标注dimension_id；某维度没有直接证据时不生成Claim，改在dimension_status标记insufficient。
10. 每个Claim只引用直接支持它的证据，通常1到4条、最多5条；仅主题相关不能算支持。
11. evidence_ids按直接支持程度排序，优先一手论文、官方资料和高权威来源。
12. 禁止用一个宽泛Claim概括用户要求的全部子问题，也禁止让一个Claim引用几乎全部Evidence。

只输出JSON：

{{
  "draft_answer": "当前研究结论",
  "claims": [
    {{
      "claim_id": "C1",
      "dimension_id": "D1",
      "claim": "一个关键判断",
      "evidence_ids": ["E1", "E2"],
      "confidence": 0.85
    }},
    {{
      "claim_id": "C2",
      "dimension_id": "D2",
      "claim": "另一个关键判断",
      "evidence_ids": ["E3"],
      "confidence": 0.70
    }}
  ],
  "dimension_status": [
    {{"dimension_id": "D1", "status": "supported"}},
    {{"dimension_id": "D2", "status": "partial"}}
  ],
  "uncertainties": [
    "当前仍然无法确认的问题"
  ]
}}

claim_id必须依次使用C1、C2、C3。
dimension_id必须来自给定analysis_dimensions。
dimension_status只能使用supported、partial、insufficient。
evidence_ids只能引用真实Evidence，且每个Claim最多5条。"""

        result = self._parse_json(
            self.llm.generate(prompt, max_new_tokens=2200)
        )

        if result is None:
            return {
                "draft_answer": "Researcher未能生成结构化研究结果。",
                "claims": [],
                "dimension_status": [
                    {
                        "dimension_id": item.get("id", ""),
                        "status": "insufficient"
                    }
                    for item in analysis_dimensions
                    if isinstance(item, dict)
                ],
                "uncertainties": ["Researcher输出解析失败"]
            }

        claims = result.get(
            "claims",
            []
        )

        if not isinstance(claims, list):
            claims = []

        valid_dimension_ids = [
            str(item.get("id", ""))
            for item in analysis_dimensions
            if isinstance(item, dict) and item.get("id")
        ]
        default_dimension_id = valid_dimension_ids[0] if valid_dimension_ids else "D1"
        normalized_claims = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim") or "").strip()
            if not claim_text:
                continue

            dimension_id = str(claim.get("dimension_id") or "").upper()
            if dimension_id not in valid_dimension_ids:
                dimension_id = default_dimension_id

            raw_evidence_ids = claim.get("evidence_ids", [])
            if not isinstance(raw_evidence_ids, list):
                raw_evidence_ids = [raw_evidence_ids]

            evidence_ids = []
            for evidence_id in raw_evidence_ids:
                match = re.fullmatch(r"E([1-9]\d*)", str(evidence_id).strip().upper())
                if not match or int(match.group(1)) > len(evidence):
                    continue
                normalized_id = f"E{int(match.group(1))}"
                if normalized_id not in evidence_ids:
                    evidence_ids.append(normalized_id)
                if len(evidence_ids) >= 5:
                    break

            normalized_claim = dict(claim)
            normalized_claim["claim_id"] = f"C{len(normalized_claims) + 1}"
            normalized_claim["dimension_id"] = dimension_id
            normalized_claim["claim"] = claim_text
            normalized_claim["evidence_ids"] = evidence_ids
            normalized_claims.append(normalized_claim)

        result["claims"] = normalized_claims

        raw_statuses = result.get("dimension_status", [])
        if not isinstance(raw_statuses, list):
            raw_statuses = []
        status_by_dimension = {}
        for item in raw_statuses:
            if not isinstance(item, dict):
                continue
            dimension_id = str(item.get("dimension_id") or "").upper()
            status = str(item.get("status") or "").lower()
            if dimension_id in valid_dimension_ids and status in {
                "supported", "partial", "insufficient"
            }:
                status_by_dimension[dimension_id] = status

        covered_dimensions = {
            claim["dimension_id"]
            for claim in normalized_claims
            if claim.get("evidence_ids")
        }
        result["dimension_status"] = [
            {
                "dimension_id": dimension_id,
                "status": status_by_dimension.get(
                    dimension_id,
                    "partial" if dimension_id in covered_dimensions else "insufficient"
                )
            }
            for dimension_id in valid_dimension_ids
        ]

        return result

    @staticmethod
    def _sanitize_answer(answer, evidence_count):
        # 删除模型自行生成的参考文献/证据来源部分。
        pattern = (
            r"\n\s*(?:#{1,6}\s*)?"
            r"(?:参考文献|References|证据来源|参考资料)"
            r"\s*:?\s*\n"
        )
        answer = re.split(
            pattern,
            answer,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        # 删除旧式 [1]、[2] 引用，避免与 Evidence ID 混淆。
        answer = re.sub(r"\[(\d+)\]", "", answer)

        # 删除不存在的 [E99] 等引用。
        def replace_invalid(match):
            evidence_id = int(match.group(1))
            if 1 <= evidence_id <= evidence_count:
                return match.group(0)
            return ""

        return re.sub(
            r"\[E(\d+)\]",
            replace_invalid,
            answer
        ).strip()

    @staticmethod
    def _append_verified_sources(answer, evidence):
        cited_ids = {
            int(x)
            for x in re.findall(r"\[E(\d+)\]", answer)
        }

        if not cited_ids:
            return answer

        lines = ["", "### 证据来源"]

        for evidence_id in sorted(cited_ids):
            item = evidence[evidence_id - 1]

            title = item.get("title") or item.get(
                "source_name",
                "Unknown Source"
            )
            source_type = item.get("source_type", "")
            url = item.get("url", "")

            line = f"[E{evidence_id}] [{source_type}] {title}"

            if url:
                line += f" — {url}"

            lines.append(line)

        return answer + "\n" + "\n".join(lines)

    def finalize(
        self,
        question,
        analysis_dimensions,
        verified_claims,
        dimension_status,
        critique,
        unsupported_claims,
        missing_perspectives,
        evidence_conflicts,
        sufficient,
        evidence
    ):
        def clean_text(value):
            return re.sub(
                r"\[E\d+\]",
                "",
                str(value or "")
            ).strip()

        unsupported_texts = []
        for item in unsupported_claims or []:
            text = clean_text(item)
            if text and text not in unsupported_texts:
                unsupported_texts.append(text)

        gaps = []
        for item in missing_perspectives or []:
            if isinstance(item, dict):
                item = (
                    item.get("perspective")
                    or item.get("text")
                    or item.get("reason")
                    or item.get("query")
                    or ""
                )
            text = clean_text(item)
            if text and text not in gaps:
                gaps.append(text)

        dimensions = []
        for index, item in enumerate(analysis_dimensions or [], 1):
            if not isinstance(item, dict):
                continue
            dimension_id = str(item.get("id") or f"D{index}").upper()
            dimension_question = clean_text(item.get("question"))
            if dimension_question:
                dimensions.append({
                    "id": dimension_id,
                    "question": dimension_question
                })
        if not dimensions:
            dimensions = [{"id": "D1", "question": question}]

        status_by_dimension = {
            str(item.get("dimension_id", "")).upper(): str(
                item.get("status", "insufficient")
            ).lower()
            for item in dimension_status or []
            if isinstance(item, dict)
        }
        claims_by_dimension = {item["id"]: [] for item in dimensions}
        unmatched_claims = []
        for claim in verified_claims or []:
            if not isinstance(claim, dict):
                continue
            dimension_id = str(claim.get("dimension_id") or "").upper()
            if dimension_id in claims_by_dimension:
                claims_by_dimension[dimension_id].append(claim)
            else:
                unmatched_claims.append(claim)

        def claim_text_with_citations(claim):
            text = clean_text(claim.get("claim"))
            citations = " ".join(
                f"[{evidence_id}]"
                for evidence_id in claim.get("evidence_ids", [])
            )
            if claim.get("verification_status") == "partial":
                text = f"现有证据有限支持：{text}"
            return f"{text} {citations}".rstrip()

        lines = ["### 核心结论", ""]
        if verified_claims:
            summary = "；".join(
                claim_text_with_citations(claim).rstrip("。")
                for claim in verified_claims[:3]
                if isinstance(claim, dict)
            )
            if summary:
                lines.append(summary + "。")
        else:
            lines.append("当前没有通过逐Claim证据核验的事实性结论。")
        if not sufficient:
            lines.append("Critic判断当前整体证据仍不充分，因此以下结论不能外推为普遍规律。")

        for index, dimension in enumerate(dimensions, 1):
            dimension_id = dimension["id"]
            lines.extend(["", f"### {index}. {dimension['question']}", ""])
            dimension_claims = claims_by_dimension.get(dimension_id, [])
            if dimension_claims:
                for claim in dimension_claims:
                    lines.append(f"- {claim_text_with_citations(claim)}")
                continue

            question_text = dimension["question"].lower()
            if any(marker in question_text for marker in ("github", "开源", "工程", "实现")):
                lines.append("- 本轮没有检索到足够高质量且通过Critic核验的GitHub或工程证据。")
            elif status_by_dimension.get(dimension_id) == "partial":
                lines.append("- 当前只有部分相关证据，但没有形成通过Critic核验的原子主张。")
            else:
                lines.append("- 当前检索到的证据不足以确认这一维度的具体结论。")

        if unmatched_claims:
            lines.extend(["", "### 其他已验证发现", ""])
            for claim in unmatched_claims:
                lines.append(f"- {claim_text_with_citations(claim)}")

        if unsupported_texts:
            lines.extend(["", "### 当前仍不足以确认", ""])
            for text in unsupported_texts:
                lines.append(f"- {text}")

        if gaps:
            lines.extend(["", "### 证据缺口", ""])
            for text in gaps:
                lines.append(f"- {text}")

        if evidence_conflicts:
            lines.extend(["", "### 冲突证据", ""])
            for conflict in evidence_conflicts:
                if not isinstance(conflict, dict):
                    continue
                topic = clean_text(conflict.get("topic")) or "同一事实存在相反证据"
                citations = " ".join(
                    f"[{item}]"
                    for item in (
                        conflict.get("evidence_for", [])
                        + conflict.get("evidence_against", [])
                    )
                )
                lines.append(f"- {topic}（{conflict.get('status', 'unresolved')}） {citations}".rstrip())

        if unsupported_texts or gaps or evidence_conflicts or not sufficient:
            lines.extend(["", "### 下一步建议", ""])
            lines.append("- 如需继续研究，应优先围绕上述未覆盖维度补充直接、可核验的一手证据。")

        answer = "\n".join(lines)

        return self._append_verified_sources(
            answer,
            evidence[:12]
        )
