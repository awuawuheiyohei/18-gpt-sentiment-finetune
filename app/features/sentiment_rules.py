"""词典打分与稠密特征。

提供两套接口：

- :func:`lexicon_score` —— 纯规则打分，**baseline 用**，输出 -1 ~ +1 的连续分；
- :func:`dense_features` —— 22 维稠密特征，作为逻辑回归的附加信号。

规则打分的三个设计点：
1. **否定翻转**：情感词左侧窗口内出现否定词则极性翻转，且强度打 0.75 折
   （"不好吃" 不完全等价于 "难吃"，打折是经验取值）；
2. **程度副词加权**：左侧窗口内的程度副词对强度做倍数缩放；
3. **多字词优先**：从左到右贪心匹配，命中长词后跳过，避免重复计分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .lexicon import (
    DEGREE_ADVERBS,
    NEGATION_WINDOW,
    NEGATION_WORDS,
    SARCASM_MARKERS,
    SENTIMENT_LEXICON,
)
from .text import normalize

# 特征向量顺序 —— 训练与推理必须一致，改动需同步 bump 模型版本
DENSE_FEATURE_NAMES: Tuple[str, ...] = (
    "pos_hits",
    "neg_hits",
    "negated_hits",
    "polarity_score",
    "max_pos",
    "max_neg",
    "degree_up",
    "degree_down",
    "exclam",
    "question",
    "comma",
    "ellipsis",
    "sarcasm",
    "length_log",
    "char_len",
    "has_digit",
    "latin_char",
    "repeated_char",
    "pos_ratio",
    "neg_ratio",
    "score_abs",
    "sentence_count",
)

DENSE_FEATURE_DIM = len(DENSE_FEATURE_NAMES)

# 归一化分母，避免长文本特征无界放大
_LENGTH_CAP = 60.0


@dataclass
class LexiconHit:
    """一次词典命中的完整解释信息。"""

    term: str
    polarity: float          # 原始极性（负数=负面）
    adjusted: float          # 考虑否定与程度后的最终贡献
    negated: bool
    degree: float
    index: int               # 在归一化文本中的起始位置

    def as_dict(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "polarity": round(self.polarity, 4),
            "adjusted": round(self.adjusted, 4),
            "negated": self.negated,
            "degree": round(self.degree, 2),
            "index": self.index,
        }


def _left_window(norm: str, start: int) -> str:
    return norm[max(0, start - NEGATION_WINDOW) : start]


def _negation_in_window(window: str) -> bool:
    """判断窗口内是否存在否定词。

    ⚠️ 必须先剔除程度副词再做子串匹配，否则 "非常" 里的 "非" 会被误判成否定词，
    导致 "非常好吃" 被算成负面（这是实测踩到的坑）。
    """
    residue = window
    for adv in DEGREE_ADVERBS:
        if adv in residue:
            residue = residue.replace(adv, "")
    return any(neg in residue for neg in NEGATION_WORDS)


def scan_hits(norm_text: str) -> List[LexiconHit]:
    """在归一化文本上扫描全部情感词命中。"""
    if not norm_text:
        return []
    hits: List[LexiconHit] = []
    i = 0
    n = len(norm_text)
    while i < n:
        matched = None
        for size in range(6, 0, -1):
            if i + size > n:
                continue
            cand = norm_text[i : i + size]
            if cand in SENTIMENT_LEXICON:
                matched = (cand, size)
                break
        if matched is None:
            i += 1
            continue
        term, size = matched
        window = _left_window(norm_text, i)
        negated = _negation_in_window(window)
        # 程度副词：取窗口内倍数最大的一个，没有则 1.0
        degree = max(
            (mult for adv, mult in DEGREE_ADVERBS.items() if adv in window),
            default=1.0,
        )
        polarity = SENTIMENT_LEXICON[term]
        adjusted = polarity * degree
        if negated:
            adjusted = -adjusted * 0.75
        hits.append(
            LexiconHit(
                term=term,
                polarity=polarity,
                adjusted=adjusted,
                negated=negated,
                degree=degree,
                index=i,
            )
        )
        i += size
    return hits


def lexicon_score(text: str) -> Tuple[float, List[LexiconHit]]:
    """纯词典打分。

    返回 (score, hits)。score ∈ [-1, 1]：>0 正面，<0 负面，==0 无法判断。
    打分用 tanh 压缩，长文本不会因为命中多而无限增大。
    """
    import math

    norm = normalize(text)
    hits = scan_hits(norm)
    if not hits:
        return 0.0, []
    raw = sum(h.adjusted for h in hits)
    # 按命中数归一，避免"堆词"放大信号
    avg = raw / math.sqrt(len(hits))
    score = math.tanh(avg)
    return round(score, 6), hits


def dense_features(text: str) -> List[float]:
    """22 维稠密特征。顺序见 :data:`DENSE_FEATURE_NAMES`。"""
    norm = normalize(text)
    score, hits = lexicon_score(text)

    pos_hits = sum(1 for h in hits if h.polarity > 0)
    neg_hits = sum(1 for h in hits if h.polarity < 0)
    negated_hits = sum(1 for h in hits if h.negated)
    max_pos = max([h.adjusted for h in hits if h.polarity > 0], default=0.0)
    max_neg = min([h.adjusted for h in hits if h.polarity < 0], default=0.0)
    degree_up = sum(h.degree for h in hits if h.polarity > 0 and h.degree > 1.0)
    degree_down = sum(1.0 for h in hits if h.polarity > 0 and h.degree < 1.0)

    exclam = norm.count("!")
    question = norm.count("?")
    comma = norm.count(",") + norm.count(".")
    ellipsis = norm.count("...") + norm.count("…")
    sarcasm = float(any(m in norm for m in SARCASM_MARKERS))

    length_log = _safe_log(len(norm))
    total_hits = pos_hits + neg_hits
    repeated = float(_max_repeat_run(norm) >= 2)
    latin = sum(1 for ch in norm if ch.isascii() and ch.isalpha())

    return [
        float(pos_hits),
        float(neg_hits),
        float(negated_hits),
        float(score),
        float(max_pos),
        float(max_neg),
        float(degree_up),
        float(degree_down),
        float(exclam),
        float(question),
        float(comma),
        float(ellipsis),
        sarcasm,
        length_log,
        float(len(norm)),
        float(any(ch.isdigit() for ch in norm)),
        float(latin),
        repeated,
        float(pos_hits) / total_hits if total_hits else 0.0,
        float(neg_hits) / total_hits if total_hits else 0.0,
        abs(score),
        float(sum(1 for ch in norm if ch in ",.!?;") + 1),
    ]


def _safe_log(x: float) -> float:
    import math

    return math.log1p(min(float(x), _LENGTH_CAP))


def _max_repeat_run(text: str, n: int = 2) -> int:
    """某个 n-gram 片段在全文中最多重复出现的次数。

    复读是外卖评论里很强的负面信号（"没有送水没有送水没有送水"、"送了三次都没到"），
    所以这里检测的是**片段重复**而不是连续重复字符。
    """
    if len(text) < n:
        return 0
    counts: Dict[str, int] = {}
    for i in range(len(text) - n + 1):
        gram = text[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) if counts else 0


def lexicon_baseline_predict(text: str, threshold: float = 0.0) -> int:
    """纯规则 baseline 的预测：score > threshold -> 正面(1)，否则负面(0)。

    threshold=0 时等价于"有情感词且净得分为正才判正面"，对不平衡数据偏保守，
    因此 macro-F1 反而可能不错——这正是要实测的点。
    """
    score, _ = lexicon_score(text)
    return 1 if score > threshold else 0