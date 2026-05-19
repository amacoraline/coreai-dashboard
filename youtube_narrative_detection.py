import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer,
)

from sklearn.cluster import (
    MiniBatchKMeans,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

from sklearn.decomposition import (
    LatentDirichletAllocation,
)


# =====================================
# LOAD MODEL
# =====================================

print(
    "\nLoading sentence transformer model..."
)

embedding_model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2"
)

print(
    "Sentence transformer loaded.\n"
)


# =====================================
# CONFIG
# =====================================

NARRATIVE_OUTPUT_FILE = (
    "youtube_narrative_detection.csv"
)

TOPIC_COUNT = 10

SCORE_RANGE_THRESHOLDS = {
    "viral_probability":          (0.40, 0.70),
    "toxicity_score":             (0.30, 0.60),
    "sentiment_score":            (0.40, 0.70),
    "emotion_score":              (0.40, 0.70),
    "narrative_similarity_score": (0.40, 0.70),
    "virality_score":             (0.40, 0.70),
    "engagement_rate":            (0.02, 0.05),
    "pharma_risk_score":          (1,    3   ),
    "cluster_volume":             (10,   30  ),
    "views_per_hour":             (100,  1000),
    "likes_per_hour":             (10,   100 ),
    "comments_per_hour":          (5,    50  ),
    "view_velocity":              (100,  1000),
    "like_velocity":              (10,   100 ),
    "comment_velocity":           (5,    50  ),
    "growth_acceleration":        (0,    500 ),
    "trend_score":                (0.30, 0.70),
    "narrative_trend_score":      (5,    15  ),
}


def add_score_range_labels(df):

    for col, (low_thresh, high_thresh) in (
        SCORE_RANGE_THRESHOLDS.items()
    ):

        if col not in df.columns:
            continue

        series = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

        df[f"{col}_range"] = np.select(
            [
                series < low_thresh,
                (series >= low_thresh) & (series < high_thresh),
                series >= high_thresh,
            ],
            ["Low", "Medium", "High"],
            default="Low",
        )

    return df


# =====================================
# UTILITY FUNCTIONS
# =====================================

def load_dataset(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    print(
        f"\nLoading dataset:\n{path}"
    )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def find_latest_tracking_file():

    current_dir = Path.cwd()

    files = list(
        current_dir.glob(
            "youtube_tracking_features.csv"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No youtube_tracking_features.csv found."
        )

    latest_file = max(
        files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    return latest_file


def clean_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip()


# =====================================
# EMBEDDINGS
# =====================================

def generate_embeddings(
    text_list,
):

    print(
        "\nGenerating embeddings..."
    )

    embeddings = embedding_model.encode(
        text_list,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


# =====================================
# CLUSTERING
# =====================================

def cluster_narratives(
    embeddings,
):

    print(
        "\nClustering narratives..."
    )

    cluster_count = min(
        50,
        max(
            5,
            len(embeddings) // 1000
        )
    )

    print(
        f"Using {cluster_count} clusters..."
    )

    clustering_model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        batch_size=1024,
        n_init="auto",
    )

    cluster_labels = (
        clustering_model.fit_predict(
            embeddings
        )
    )

    return cluster_labels


# =====================================
# TOPIC MODELING
# =====================================

def extract_topics(
    text_list,
):

    print(
        "\nExtracting topics..."
    )

    cleaned_texts = []

    for text in text_list:

        if pd.isna(text):
            continue

        text = str(text).strip()

        if len(text) < 10:
            continue

        cleaned_texts.append(text)

    # FALLBACK
    if len(cleaned_texts) == 0:

        print(
            "\nNo valid text found for topic modeling."
        )

        dominant_topics = [0] * len(text_list)

        topic_keywords = {
            0: "no_topic_detected"
        }

        return (
            dominant_topics,
            topic_keywords,
        )

    try:

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=2000,
            min_df=1,
        )

        document_term_matrix = (
            vectorizer.fit_transform(
                cleaned_texts
            )
        )

        # EMPTY VOCAB CHECK
        if (
            document_term_matrix.shape[1]
            == 0
        ):

            print(
                "\nEmpty vocabulary after vectorization."
            )

            dominant_topics = [0] * len(text_list)

            topic_keywords = {
                0: "empty_topic"
            }

            return (
                dominant_topics,
                topic_keywords,
            )

        n_topics = min(
            TOPIC_COUNT,
            max(3, len(cleaned_texts) // 20),
            document_term_matrix.shape[1],
        )

        lda_model = (
            LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
            )
        )

        lda_model.fit(
            document_term_matrix
        )

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        topic_keywords = {}

        for topic_idx, topic in enumerate(
            lda_model.components_
        ):

            top_words = [
                feature_names[i]
                for i in topic.argsort()[-10:]
            ]

            topic_keywords[
                topic_idx
            ] = ", ".join(top_words)

        topic_distribution = (
            lda_model.transform(
                document_term_matrix
            )
        )

        dominant_topics_small = np.argmax(
            topic_distribution,
            axis=1,
        )

        # PAD BACK TO ORIGINAL SIZE
        dominant_topics = []

        current_index = 0

        for text in text_list:

            if (
                pd.isna(text)
                or len(str(text).strip()) < 10
            ):

                dominant_topics.append(0)

            else:

                dominant_topics.append(
                    int(
                        dominant_topics_small[
                            current_index
                        ]
                    )
                )

                current_index += 1

        return (
            dominant_topics,
            topic_keywords,
        )

    except Exception as error:

        print(
            "\nTopic modeling failed:"
        )

        print(str(error))

        dominant_topics = [0] * len(text_list)

        topic_keywords = {
            0: "topic_model_failed"
        }

        return (
            dominant_topics,
            topic_keywords,
        )


# =====================================
# NARRATIVE TOPIC DEFINITIONS
# =====================================

NARRATIVE_TOPIC_DEFINITIONS = {
    "Off-label Claims": [
        "off label", "off-label", "unapproved use", "unlicensed",
        "unauthorized use", "not approved", "non approved",
        "used for", "prescribed for", "treating with",
    ],
    "Safety & Side Effects": [
        "side effect", "side effects", "adverse event", "adverse reaction",
        "adverse effects", "reaction", "harm", "injury", "injured",
        "complication", "complications", "negative effect",
    ],
    "Clinical Trial Results": [
        "clinical trial", "phase 1", "phase 2", "phase 3", "phase 4",
        "placebo", "efficacy", "randomized", "double blind", "study results",
        "trial results", "research study", "participants", "endpoint",
    ],
    "Patient Experience": [
        "my experience", "my story", "personal story", "patient", "patients",
        "living with", "diagnosed", "my doctor", "i was prescribed",
        "i took", "i started", "how i", "my journey", "testimonial",
    ],
    "Regulatory & Legal Action": [
        "fda warning", "ema warning", "recall", "market withdrawal",
        "lawsuit", "litigation", "class action", "safety alert",
        "black box warning", "regulatory action", "pharmacovigilance",
        "post market", "compliance", "enforcement",
    ],
    "Pharma Safety Alert": [
        "death", "died", "fatal", "fatality", "hospitalized",
        "hospitalisation", "emergency", "serious injury", "dangerous",
        "unsafe", "hazardous", "toxic", "toxicity", "overdose",
    ],
    "Misinformation & False Claims": [
        "fake cure", "miracle", "cure all", "conspiracy", "fake news",
        "misinformation", "natural cure", "big pharma lies",
        "alternative medicine", "homeopathy", "detox", "scam",
    ],
    "Competitor & Brand Comparison": [
        "abbvie", "sanofi", "regeneron", "bayer", "eli lilly",
        "compared to", "better than", "switch from", "versus",
        "alternative to", "competitor", "biosimilar switch",
    ],
    "Brand Sentiment & Reviews": [
        "brand", "reputation", "opinion", "review", "rating",
        "trust", "recommend", "disappointed", "satisfied",
        "happy with", "unhappy", "works well", "does not work",
    ],
    "Contraception & IUD": [
        "nuvaring", "nexplanon", "mirena", "kyleena", "implanon",
        "iud", "implant", "birth control", "contraception",
        "paragard", "intrauterine", "hormonal contraception",
    ],
    "Fertility & IVF": [
        "ivf", "fertility", "infertility", "embryo", "egg freezing",
        "ovarian stimulation", "gonadotropin", "puregon", "gonal",
        "embryo transfer", "fertility treatment", "ovulation induction",
    ],
    "Dermatology & Psoriasis": [
        "psoriasis", "eczema", "atopic dermatitis", "dupixent",
        "skin rash", "plaque", "itching", "pruritus", "vtama",
        "tapinarof", "skin condition", "dermatology",
    ],
    "Immunology & Biologics": [
        "biosimilar", "biologic", "humira", "adalimumab", "renflexis",
        "rheumatoid arthritis", "crohn", "ulcerative colitis",
        "ankylosing spondylitis", "rinvoq", "biologic therapy",
    ],
    "Respiratory & Asthma": [
        "asthma", "singulair", "montelukast", "allergy", "inhaler",
        "bronchospasm", "nasonex", "dulera", "allergic rhinitis",
        "respiratory", "breathing", "eosinophilic",
    ],
    "Migraine Treatment": [
        "migraine", "headache", "maxalt", "rizatriptan", "cgrp",
        "emgality", "rayvow", "acute migraine", "chronic migraine",
        "migraine attack", "prevention",
    ],
    "Hair Loss Treatment": [
        "hair loss", "propecia", "finasteride", "alopecia",
        "baldness", "thinning hair", "hairfall", "male pattern",
        "androgenetic",
    ],
    "Women's Health & Menopause": [
        "menopause", "hot flashes", "hrt", "hormone replacement",
        "vaginal", "duavive", "prometrium", "xaciato",
        "perimenopause", "estrogen", "progesterone",
    ],
    "Drug Pricing & Access": [
        "price", "cost", "afford", "insurance", "coverage",
        "generic", "expensive", "copay", "out of pocket",
        "drug access", "healthcare cost", "unaffordable",
    ],
    "Influencer & Sponsored Content": [
        "influencer", "sponsored", "paid partnership", "ad",
        "ambassador", "gifted", "collab", "promotion",
        "social media", "instagram", "tiktok", "youtube channel",
    ],
    "Treatment & Dosage Discussion": [
        "dosage", "dose", "prescription", "prescribed", "mg",
        "treatment plan", "medication", "therapy",
        "how to take", "when to take", "injection", "tablet",
    ],
}


# =====================================
# CLUSTER TOPIC NAMING
# =====================================

def generate_cluster_topic_names(df):

    print(
        "\nGenerating cluster topic names..."
    )

    cluster_texts = (
        df.groupby("narrative_cluster_id")[
            "analysis_text_clean"
        ]
        .apply(
            lambda texts: " ".join(
                texts.fillna("").astype(str)
            )
        )
        .to_dict()
    )

    topic_names = {}

    for cid, text in cluster_texts.items():

        text_lower = text.lower()

        scores = {}

        for topic_name, keywords in (
            NARRATIVE_TOPIC_DEFINITIONS.items()
        ):
            score = sum(
                1 for kw in keywords
                if kw in text_lower
            )
            if score > 0:
                scores[topic_name] = score

        if scores:
            topic_names[cid] = max(
                scores, key=scores.get
            )
        else:
            # Fallback: TF-IDF top terms for unmatched clusters
            topic_names[cid] = _tfidf_fallback(
                text_lower, cid
            )

    return topic_names


def _tfidf_fallback(text, cid):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2),
        )

        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        row = tfidf_matrix[0].toarray().flatten()
        top_indices = row.argsort()[-3:][::-1]

        terms = [
            feature_names[i].title()
            for i in top_indices
            if row[i] > 0
        ]

        return " ".join(terms) if terms else f"Cluster {cid}"

    except Exception:

        return f"Cluster {cid}"


# =====================================
# SIMILARITY
# =====================================

def calculate_similarity_scores(
    embeddings,
):

    print(
        "\nCalculating similarity scores..."
    )

    sample_size = min(
        1000,
        len(embeddings)
    )

    sampled_embeddings = embeddings[
        :sample_size
    ]

    similarity_matrix = cosine_similarity(
        sampled_embeddings
    )

    max_similarity_scores = []

    for index in range(
        sample_size
    ):

        similarities = np.delete(
            similarity_matrix[index],
            index,
        )

        if len(similarities) == 0:

            max_similarity_scores.append(
                0
            )

        else:

            max_similarity_scores.append(
                round(
                    np.max(similarities),
                    4,
                )
            )

    # PAD REMAINING
    if len(embeddings) > sample_size:

        remaining = (
            len(embeddings)
            - sample_size
        )

        max_similarity_scores.extend(
            [0] * remaining
        )

    return max_similarity_scores


# =====================================
# NARRATIVE TRENDING
# =====================================

def calculate_narrative_trends(
    df,
):

    print(
        "\nCalculating narrative trends..."
    )

    narrative_counts = (
        df.groupby(
            "narrative_cluster_id"
        )
        .size()
        .reset_index(
            name="cluster_volume"
        )
    )

    df = df.merge(
        narrative_counts,
        on="narrative_cluster_id",
        how="left",
    )

    df["narrative_trend_score"] = (
        (
            df["cluster_volume"]
            .fillna(0)
            * 0.4
        )
        +
        (
            df["virality_score"]
            .fillna(0)
            * 0.3
        )
        +
        (
            df["trend_score"]
            .fillna(0)
            * 0.3
        )
    )

    threshold = (
        df[
            "narrative_trend_score"
        ]
        .quantile(0.95)
    )

    df["emerging_narrative"] = np.where(
        df[
            "narrative_trend_score"
        ] >= threshold,
        "emerging",
        "normal",
    )

    return df


# =====================================
# PIPELINE
# =====================================

def build_narrative_pipeline(
    df,
):

    if (
        "analysis_text_clean"
        not in df.columns
    ):

        if (
            "analysis_text_en"
            in df.columns
        ):

            df[
                "analysis_text_clean"
            ] = (
                df[
                    "analysis_text_en"
                ]
                .fillna("")
                .astype(str)
            )

        else:

            raise ValueError(
                "analysis_text_clean column missing."
            )

    df["analysis_text_clean"] = (
        df["analysis_text_clean"]
        .fillna("")
        .astype(str)
    )

    # EMBEDDINGS
    embeddings = generate_embeddings(
        df["analysis_text_clean"]
        .tolist()
    )

    # CLUSTERING
    cluster_labels = (
        cluster_narratives(
            embeddings
        )
    )

    df["narrative_cluster_id"] = (
        cluster_labels
    )

    # TOPICS
    (
        dominant_topics,
        topic_keywords,
    ) = extract_topics(
        df["analysis_text_clean"]
        .tolist()
    )

    df["dominant_topic_id"] = (
        dominant_topics
    )

    df["topic_keywords"] = (
        df["dominant_topic_id"]
        .map(topic_keywords)
        .fillna(
            df.get("keyword", pd.Series("unclassified", index=df.index))
            .fillna("unclassified")
        )
    )

    # CLUSTER TOPIC NAME — TF-IDF distinctive terms per cluster
    cluster_topic_map = generate_cluster_topic_names(df)

    df["cluster_topic_name"] = (
        df["narrative_cluster_id"]
        .map(cluster_topic_map)
        .fillna("Unclassified")
    )

    # SIMILARITY
    df["narrative_similarity_score"] = (
        calculate_similarity_scores(
            embeddings
        )
    )

    # TRENDING
    df = calculate_narrative_trends(
        df
    )

    # SCORE RANGE LABELS
    df = add_score_range_labels(df)

    return df


# =====================================
# SAVE OUTPUTS
# =====================================

def save_outputs(df):

    # CSV
    df.to_csv(
        NARRATIVE_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved narrative CSV:\n"
        f"{NARRATIVE_OUTPUT_FILE}"
    )

    # PARQUET
    try:

        df.to_parquet(
            "youtube_narrative_detection.parquet",
            index=False,
        )

        print(
            "\nSaved narrative parquet."
        )

    except Exception as error:

        print(
            "\nParquet save failed:"
        )

        print(str(error))


# =====================================
# MAIN
# =====================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YouTube Narrative Detection"
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional tracking features CSV"
        ),
    )

    args = parser.parse_args()

    # FIND FILE
    if args.input:

        input_path = Path(
            args.input
        )

    else:

        print(
            "\nSearching for "
            "youtube_tracking_features.csv..."
        )

        input_path = (
            find_latest_tracking_file()
        )

    print(
        f"\nUsing file:\n"
        f"{input_path}"
    )

    # LOAD DATA
    df = load_dataset(
        input_path
    )

    print(
        f"\nLoaded rows: {len(df)}"
    )

    # PIPELINE
    narrative_df = (
        build_narrative_pipeline(
            df
        )
    )

    # SAVE
    save_outputs(
        narrative_df
    )

    print(
        "\nNarrative detection completed successfully."
    )


if __name__ == "__main__":
    main()