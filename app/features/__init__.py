"""特征工程子包：文本归一化、词典、稠密特征、稀疏 n-gram 管线。"""

from .lexicon import (
    DEGREE_ADVERBS,
    NEGATION_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    SENTIMENT_LEXICON,
    lexicon_size,
    lexicon_terms,
)
from .pipeline import (
    DENSE_FEATURE_DIM,
    DENSE_FEATURE_NAMES,
    Explanation,
    FeaturePipeline,
    FeatureSpec,
    build_pipeline,
)
from .sentiment_rules import (
    LexiconHit,
    dense_features,
    lexicon_baseline_predict,
    lexicon_score,
)
from .text import (
    char_ngrams,
    has_content,
    keyword_spans,
    ngram_counts,
    normalize,
    tokenize,
)

__all__ = [
    "DEGREE_ADVERBS",
    "NEGATION_WORDS",
    "NEGATIVE_WORDS",
    "POSITIVE_WORDS",
    "SENTIMENT_LEXICON",
    "lexicon_size",
    "lexicon_terms",
    "DENSE_FEATURE_DIM",
    "DENSE_FEATURE_NAMES",
    "Explanation",
    "FeaturePipeline",
    "FeatureSpec",
    "build_pipeline",
    "LexiconHit",
    "dense_features",
    "lexicon_baseline_predict",
    "lexicon_score",
    "char_ngrams",
    "has_content",
    "keyword_spans",
    "ngram_counts",
    "normalize",
    "tokenize",
]