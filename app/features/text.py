"""中文文本归一化与字符 n-gram 特征。

为什么用字符 n-gram 而不是分词？
--------------------------------
中文没有天然词边界，jieba 等分词器在**短文本 + 领域外送评论**上容易切错
（"不好吃" 可能被切成 "不" + "好吃"，否定范围就丢了）。字符 n-gram：

1. 不依赖词典与分词器，训练/推理链路无额外模型依赖，便于复现与跨环境部署；
2. 对未登录词、错别字、方言表达天然鲁棒（外卖评论里错别字很多）；
3. "不好吃"/"很好吃" 的差异天然落在 2~4 gram 上，无需显式否定规则。

代价是特征维度更高、序列更长；在这份平均 25 字的短评论上，实测收益为正
（见 reports/evaluation.json 中 n-gram 范围消融）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

# 全角 -> 半角需要处理偏移的标点区
_PUNCT_MAP = {
    "！": "!", "？": "?", "，": ",", "。": ".", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "、": ",", "～": "~", "－": "-", "＋": "+",
}

_SPACE_RE = re.compile(r"\s+")
_KEPT_PUNCT = set(",.!?;:~")


def normalize(text: str) -> str:
    """文本归一化。

    处理顺序（顺序本身影响特征，一经确定就不再改）：
    1. NFKC 兼容分解，统一全角/半角、兼容汉字；
    2. 显式替换中文标点为半角，保证 "!"/"?" 这类强情感符号不丢；
    3. 连续空白压成单个空格；
    4. 只保留中英文、数字与少量标点，其余（表情、emoji 等）剔除。
    """
    if not text:
        return ""
    # NFKC 会把全角 ａ 等折叠成半角，中文标点不在其中，需单独映射
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    text = _SPACE_RE.sub(" ", text.strip())

    buf: List[str] = []
    for ch in text:
        if ch in _KEPT_PUNCT:
            buf.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":  # CJK 统一表意文字
            buf.append(ch)
        elif ch.isascii() and (ch.isalnum() or ch == " "):
            buf.append(ch)
        # 其余（emoji、全角符号、零宽字符）丢弃
    return "".join(buf).strip()


def has_content(normalized: str) -> bool:
    """归一化后是否还含有**有意义的字符**（汉字 / 字母 / 数字）。

    归一化会保留 ``!`` ``?`` 等标点，所以 "！！！" 归一化后非空，
    但它显然不含任何情感信息，必须单独挡掉，否则模型会给出一个无意义的置信度。
    """
    return any(
        ("\u4e00" <= ch <= "\u9fff") or (ch.isascii() and ch.isalnum())
        for ch in normalized
    )


def char_ngrams(text: str, n_min: int = 1, n_max: int = 4) -> List[str]:
    """提取字符 n-gram。已归一化输入。"""
    grams: List[str] = []
    length = len(text)
    for n in range(n_min, n_max + 1):
        if length < n:
            continue
        grams.extend(text[i : i + n] for i in range(length - n + 1))
    return grams


def ngram_counts(text: str, n_min: int = 1, n_max: int = 4) -> Dict[str, int]:
    """字符 n-gram 计数。用于解释接口展示"重要片段"。"""
    counts: Dict[str, int] = {}
    for gram in char_ngrams(text, n_min, n_max):
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def tokenize(text: str) -> List[str]:
    """把归一化文本切成"片段"：中文按连续汉字串切，数字/英文整体成词。

    仅用于词典匹配与解释展示，不参与模型特征（模型用字符 n-gram）。
    """
    text = normalize(text)
    if not text:
        return []
    tokens: List[str] = []
    cjk: List[str] = []
    latin: List[str] = []

    def flush() -> None:
        if cjk:
            tokens.append("".join(cjk))
            cjk.clear()
        if latin:
            tokens.append("".join(latin))
            latin.clear()

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if latin:
                tokens.append("".join(latin))
                latin.clear()
            cjk.append(ch)
        elif ch.isascii() and ch.isalnum():
            if cjk:
                tokens.append("".join(cjk))
                cjk.clear()
            latin.append(ch)
        else:
            # 标点/空白同时结束两个缓冲
            flush()
    flush()
    return tokens


def keyword_spans(text: str, terms: Tuple[str, ...] | List[str]) -> List[Tuple[str, int]]:
    """在**归一化后的**文本里定位词典词，返回 (词, 起始下标)。

    贪心从左到右匹配，命中即跳过，避免 "好吃" 与 "好吃吗" 重复计分。
    """
    norm = normalize(text)
    term_set = set(terms)
    spans: List[Tuple[str, int]] = []
    i = 0
    while i < len(norm):
        hit = None
        # 最多看 6 个字，外卖评论情感词很少超过 4 字
        for size in range(6, 0, -1):
            cand = norm[i : i + size]
            if cand in term_set:
                hit = cand
                break
        if hit:
            spans.append((hit, i))
            i += len(hit)
        else:
            i += 1
    return spans