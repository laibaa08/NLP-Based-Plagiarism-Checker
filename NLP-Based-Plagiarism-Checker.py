import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# PREPROCESSING
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(w) for w in tokens if len(w) > 1]
    return tokens

# TERM FREQUENCY
def calculate_tf(words):
    tf = {}
    total = len(words)

    for w in words:
        tf[w] = tf.get(w, 0) + 1

    for w in tf:
        tf[w] /= total

    return tf

# INVERSE DOCUMENT FREQUENCY
def calculate_idf(documents):
    idf = {}
    total_docs = len(documents)
    vocab = set()

    for doc in documents:
        vocab.update(doc)

    for word in vocab:
        count = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((total_docs + 1) / (count + 1)) + 1

    return idf
# TF-IDF
def calculate_tfidf(tf, idf):
    return {w: tf[w] * idf[w] for w in tf}

# VECTOR ALIGNMENT
def align_vectors(v1, v2):
    words = set(v1) | set(v2)
    return (
        {w: v1.get(w, 0) for w in words},
        {w: v2.get(w, 0) for w in words}
    )

# COSINE SIMILARITY
def cosine_similarity(v1, v2):
    dot = sum(v1[w] * v2[w] for w in v1)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

# PLAGIARISM CHECK
def plagiarism_checker_two_docs(text1, text2):
    w1 = preprocess_text(text1)
    w2 = preprocess_text(text2)

    tf1 = calculate_tf(w1)
    tf2 = calculate_tf(w2)

    idf = calculate_idf([w1, w2])

    tfidf1 = calculate_tfidf(tf1, idf)
    tfidf2 = calculate_tfidf(tf2, idf)

    tfidf1, tfidf2 = align_vectors(tfidf1, tfidf2)

    return cosine_similarity(tfidf1, tfidf2) * 100

# SENTENCE SPLIT
def split_sentences(text):
    sentences = []
    current = ""

    for ch in text:
        current += ch
        if ch in ".!?":
            sentences.append(current.strip())
            current = ""

    if current:
        sentences.append(current.strip())

    return sentences

def find_matching_sentences(text1, text2, threshold=60):
    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    matches = []

    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            sim = plagiarism_checker_two_docs(a, b)
            if sim >= threshold:
                matches.append({
                    "s1": a,
                    "s2": b,
                    "sim": sim,
                    "p1": i + 1,
                    "p2": j + 1
                })

    return matches

# PLAGIARISM REPORT
def generate_plagiarism_report(text1, text2):
    overall = plagiarism_checker_two_docs(text1, text2)
    matches = find_matching_sentences(text1, text2)

    w1 = preprocess_text(text1)
    w2 = preprocess_text(text2)
    common = set(w1) & set(w2)

    report = []
    report.append("PLAGIARISM CHECKER REPORT")

    report.append(f"\nOverall Similarity: {overall:.2f}%")

    if overall < 30:
        report.append("Verdict: No plagiarism detected")
    elif overall < 50:
        report.append("Verdict: Low similarity - likely original")
    elif overall < 70:
        report.append("Verdict: Moderate similarity - possible paraphrasing")
    else:
        report.append("Verdict: High similarity - possible plagiarism")

    report.append("DETAILED SENTENCE MATCHES:")

    if matches:
        for m in matches:
            report.append(f"\nMatch {m['p1']} -> {m['p2']}")
            report.append(f"Similarity: {m['sim']:.2f}%")
            report.append(f"Text 1: {m['s1'][:80]}...")
            report.append(f"Text 2: {m['s2'][:80]}...")
    else:
        report.append("\nNo significant sentence matches found.")

    report.append("PREPROCESSING ANALYSIS:")

    report.append(f"\nText 1 processed words ({len(w1)}):")
    report.append(f"  {', '.join(w1[:10])}{'...' if len(w1) > 10 else ''}")

    report.append(f"\nText 2 processed words ({len(w2)}):")
    report.append(f"  {', '.join(w2[:10])}{'...' if len(w2) > 10 else ''}")

    report.append(f"\nCommon words ({len(common)}):")
    report.append(f"  {', '.join(list(common)[:10])}{'...' if len(common) > 10 else ''}")

    return "\n".join(report)

# MAIN PROGRAM
if __name__ == "__main__":

    print("PLAGIARISM CHECKER")

    text1 = input("\nEnter first text:\n")
    text2 = input("\nEnter second text:\n")

    similarity = plagiarism_checker_two_docs(text1, text2)
    print(f"\nPlagiarism Similarity: {similarity:.2f}%")

    if similarity > 70:
        print(" High similarity detected ")
    elif similarity > 40:
        print(" Moderate similarity - review recommended")
    else:
        print(" Low similarity - likely original")

    choice = input("\nGenerate detailed report? (yes/no): ").lower()
    if choice == "yes":
        print("\n" + generate_plagiarism_report(text1, text2))


