import os
import re
import math
import json
import pickle
import statistics
from collections import Counter, defaultdict

from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer
import pymorphy2
from razdel import sentenize, tokenize as razdel_tokenize

MAX_LEN = 300
SAVE_PATH = "./gazeta_saved"
CACHE_DIR = "./hf_cache"

IDF_DOCS = 10000   
N = 85 

IDF_PATH = f"./gazeta_idf_{IDF_DOCS}.pkl"
LEMMA_CACHE_PATH = "./lemma_cache.pkl"

STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
    "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два",
    "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между"
}

morph = pymorphy2.MorphAnalyzer()

LEMMA_CACHE = {}

def load_lemma_cache():
    global LEMMA_CACHE
    if os.path.exists(LEMMA_CACHE_PATH):
        try:
            with open(LEMMA_CACHE_PATH, "rb") as f:
                LEMMA_CACHE = pickle.load(f)
            print(f"Лемма-кэш загружен: {len(LEMMA_CACHE)} слов")
        except Exception as e:
            print(f"Не удалось загрузить кэш лемм: {e}")
            LEMMA_CACHE = {}
    else:
        LEMMA_CACHE = {}

def save_lemma_cache():
    try:
        with open(LEMMA_CACHE_PATH, "wb") as f:
            pickle.dump(LEMMA_CACHE, f)
    except Exception as e:
        print(f"Не удалось сохранить кэш лемм: {e}")


def split_sentences(text):
    return [s.text.strip() for s in sentenize(text) if s.text.strip()]

def tokenize(text):
    tokens = []
    for t in razdel_tokenize(text):
        tok = t.text.lower()
        if re.fullmatch(r"[а-яёa-z0-9-]+", tok):
            tokens.append(tok)
    return tokens

def lemmatize_token(token):
    if token in LEMMA_CACHE:
        return LEMMA_CACHE[token]

    if token.isdigit():
        lemma = token
    else:
        parsed = morph.parse(token)
        lemma = parsed[0].normal_form if parsed else token

    LEMMA_CACHE[token] = lemma
    return lemma

def lemmatize_tokens(tokens):
    return [lemmatize_token(t) for t in tokens]

def get_content_lemmas(text):
    tokens = tokenize(text)
    lemmas = lemmatize_tokens(tokens)
    return [w for w in lemmas if w not in STOP_WORDS and len(w) > 2]

def safe_truncate(text, max_len=300):
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.7:
        return cut[:last_space].rstrip()
    return cut.rstrip()

def build_idf(train_texts, max_docs=3000):
    df = defaultdict(int)
    total_docs = min(max_docs, len(train_texts))

    for i, text in enumerate(train_texts[:total_docs]):
        if (i + 1) % 500 == 0:
            print(f"Обработано {i+1}/{total_docs} документов для IDF")

        lemmas = set(get_content_lemmas(text))
        for lemma in lemmas:
            df[lemma] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((1 + total_docs) / (1 + freq)) + 1.0

    print(f"IDF готов. Уникальных терминов: {len(idf)}")
    return idf

def load_or_build_idf(train_texts, max_docs=3000):
    path = f"./gazeta_idf_{max_docs}.pkl"

    if os.path.exists(path):
        print(f"Загружаем IDF из файла: {path}")
        with open(path, "rb") as f:
            idf = pickle.load(f)
        print(f"IDF загружен. Терминов: {len(idf)}")
        return idf

    idf = build_idf(train_texts, max_docs=max_docs)

    with open(path, "wb") as f:
        pickle.dump(idf, f)

    print(f"IDF сохранён в {path}")
    return idf

def jaccard_similarity(a_tokens, b_tokens):
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)

def overlap_ratio(a_tokens, b_tokens):
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / min(len(a_set), len(b_set))

def sentence_length_factor(sent):
    L = len(sent)
    if L < 40:
        return 0.82
    elif L <= 180:
        return 1.0
    elif L <= 260:
        return 0.94
    else:
        return 0.85

def has_quote(sent):
    return int(("«" in sent) or ("»" in sent) or ('"' in sent))

def count_digits(sent):
    return len(re.findall(r"\d", sent))

def titlecase_ratio(sent):
    words = re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+\b", sent)
    return len(words)

def get_doc_keywords(doc_lemmas, top_k=15):
    cnt = Counter(doc_lemmas)
    return [w for w, _ in cnt.most_common(top_k)]

def summarize_text(text, idf, max_len=MAX_LEN):
    sentences = split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) == 1:
        return safe_truncate(sentences[0], max_len)
    doc_lemmas = get_content_lemmas(text)

    if not doc_lemmas:
        best = min(sentences, key=lambda s: abs(len(s) - 140))
        return safe_truncate(best, max_len)
    tf = Counter(doc_lemmas)
    total_terms = sum(tf.values())

    default_idf = math.log((1 + IDF_DOCS) / 1) + 1.0
    tfidf = {}
    for term, count in tf.items():
        term_tf = count / total_terms
        term_idf = idf.get(term, default_idf)
        tfidf[term] = term_tf * term_idf

    top_keywords = [w for w, _ in Counter(doc_lemmas).most_common(20)]
    top_keywords_set = set(top_keywords)

    sent_infos = []

    sentence_lemmas_list = []
    for sent in sentences:
        sentence_lemmas_list.append(get_content_lemmas(sent))

    for i, sent in enumerate(sentences):
        lemmas = sentence_lemmas_list[i]

        if not lemmas:
            tfidf_score = 0.0
        else:
            tfidf_score = sum(tfidf.get(w, 0.0) for w in lemmas) / len(lemmas)

        pos_bonus = 1.0
        if i == 0:
            pos_bonus = 1.40
        elif i == 1:
            pos_bonus = 1.22
        elif i == 2:
            pos_bonus = 1.10
        elif i >= len(sentences) - 2:
            pos_bonus = 1.03

        len_factor = sentence_length_factor(sent)

        keyword_hits = sum(1 for w in lemmas if w in top_keywords_set)
        keyword_bonus = 1.0 + min(0.18, keyword_hits * 0.02)

        number_bonus = 1.0 + min(0.08, count_digits(sent) * 0.01)
        quote_bonus = 1.03 if has_quote(sent) else 1.0
        name_bonus = 1.0 + min(0.08, titlecase_ratio(sent) * 0.01)

        short_penalty = 0.9 if len(lemmas) <= 2 else 1.0

        base_score = (
            tfidf_score
            * pos_bonus
            * len_factor
            * keyword_bonus
            * number_bonus
            * quote_bonus
            * name_bonus
            * short_penalty
        )

        sent_infos.append({
            "idx": i,
            "text": sent,
            "lemmas": lemmas,
            "score": base_score
        })

    if all(x["score"] == 0 for x in sent_infos):
        best = min(sentences, key=lambda s: abs(len(s) - 140))
        return safe_truncate(best, max_len)

    sent_infos.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    total_len = 0

    while sent_infos:
        best_candidate = None
        best_mmr = -1e9

        for cand in sent_infos:
            relevance = cand["score"]

            redundancy = 0.0
            if selected:
                jac = max(jaccard_similarity(cand["lemmas"], sel["lemmas"]) for sel in selected)
                ov = max(overlap_ratio(cand["lemmas"], sel["lemmas"]) for sel in selected)
                redundancy = 0.6 * jac + 0.4 * ov

            mmr_score = 0.78 * relevance - 0.32 * redundancy

            if redundancy > 0.75:
                mmr_score -= 0.25

            extra = len(cand["text"]) + (1 if selected else 0)
            if total_len + extra > max_len:
                continue

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = cand

        if best_candidate is None:
            break

        selected.append(best_candidate)
        total_len += len(best_candidate["text"]) + (1 if len(selected) > 1 else 0)

        sent_infos = [x for x in sent_infos if x["idx"] != best_candidate["idx"]]

    if not selected:
        best_scored = max(
            [
                {
                    "text": s,
                    "score": 1.0 / (abs(len(s) - 140) + 1)
                }
                for s in sentences
            ],
            key=lambda x: x["score"]
        )
        return safe_truncate(best_scored["text"], max_len)

    selected.sort(key=lambda x: x["idx"])
    summary = " ".join(x["text"] for x in selected)

    return safe_truncate(summary, max_len)

def summarize_texts(texts, idf):
    return [summarize_text(text, idf) for text in texts]

def load_dataset_local():
    if os.path.exists(SAVE_PATH):
        print("Загружаем датасет")
        dataset = load_from_disk(SAVE_PATH)
    else:
        print("Скачиваем датасет")
        dataset = load_dataset(
            "IlyaGusev/gazeta",
            revision="v2.0",
            cache_dir=CACHE_DIR
        )
        dataset.save_to_disk(SAVE_PATH)
        print(f"Датасет сохранён в {SAVE_PATH}")
    return dataset

def main():
    load_lemma_cache()

    test_data = dataset["test"]
    subset = test_data.shuffle(seed=42).select(range(min(N, len(test_data))))

    texts = subset["text"]
    gold_summaries = subset["summary"]

    print(f"\nГенерируем рефераты для текстов")
    pred_summaries = summarize_texts(texts, idf)

    pred_summaries = [safe_truncate(s, 300) for s in pred_summaries]
    gold_summaries_300 = [safe_truncate(s, 300) for s in gold_summaries]

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False
    )

    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rL_p, rL_r, rL_f = [], [], []

    for gold, pred in zip(gold_summaries_300, pred_summaries):
        scores = scorer.score(gold, pred)

        r1_p.append(scores["rouge1"].precision)
        r1_r.append(scores["rouge1"].recall)
        r1_f.append(scores["rouge1"].fmeasure)

        r2_p.append(scores["rouge2"].precision)
        r2_r.append(scores["rouge2"].recall)
        r2_f.append(scores["rouge2"].fmeasure)

        rL_p.append(scores["rougeL"].precision)
        rL_r.append(scores["rougeL"].recall)
        rL_f.append(scores["rougeL"].fmeasure)

    print("\nСредние метрики")
    print(f"{'Метрика':<18} {'Precision':>10} {'Recall':>10} {'fmeasure':>10}")
    print("-" * 52)
    print(f"{'ROUGE-1':<18} {statistics.mean(r1_p):>10.4f} {statistics.mean(r1_r):>10.4f} {statistics.mean(r1_f):>10.4f}")
    print(f"{'ROUGE-2':<18} {statistics.mean(r2_p):>10.4f} {statistics.mean(r2_r):>10.4f} {statistics.mean(r2_f):>10.4f}")
    print(f"{'ROUGE-L':<18} {statistics.mean(rL_p):>10.4f} {statistics.mean(rL_r):>10.4f} {statistics.mean(rL_f):>10.4f}")

    print("\nПримеры")
    examples_shown = 0
    i = 0
    while examples_shown < 3 and i < len(texts):
        gold = gold_summaries_300[i]
        pred = pred_summaries[i]
        scores = scorer.score(gold, pred)
        if any([
            scores["rouge1"].fmeasure > 0,
            scores["rouge2"].fmeasure > 0,
            scores["rougeL"].fmeasure > 0
        ]):
            print(f"\nExample {examples_shown+1}")
            print("TEXT (первые 800 символов):")
            print(texts[i][:800], "...\n")
            print("GOLD (<=300):")
            print(gold, "\n")
            print("PRED (<=300):")
            print(pred, "\n")
            print("ROUGE-1 fmeasure: {:.4f}, ROUGE-2 fmeasure: {:.4f}, ROUGE-L fmeasure: {:.4f}".format(
                scores["rouge1"].fmeasure,
                scores["rouge2"].fmeasure,
                scores["rougeL"].fmeasure
            ))
            examples_shown += 1

        i += 1
    save_lemma_cache()


def summarize_input(text, idf):
    return summarize_text(text, idf)
if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1 — Тесты на Gazeta")
    print("2 — Ввести текст вручную")
    choice = input("Ваш выбор (1/2): ").strip()
    dataset = load_dataset_local()
    train_texts = dataset["train"]["text"]
    idf = load_or_build_idf(train_texts, max_docs=IDF_DOCS)

    if choice == "2":
        print("\nВведите текст для суммаризации (end для конца ввода):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "end":
                break
            lines.append(line)
        text = "\n".join(lines)
        summary = summarize_input(text, idf)
        print("\nSummary:")
        print(summary)
    else:
        main()