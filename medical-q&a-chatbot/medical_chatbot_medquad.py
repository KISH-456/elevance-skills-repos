"""
Medical Q&A Chatbot with MedQuAD, SBERT retrieval, simple NER, and Streamlit UI.

Requirements:
    pip install pandas sentence-transformers streamlit torch
Steps:
    1. Clone MedQuAD: https://github.com/abachaa/MedQuAD
    2. Convert / merge MedQuAD XML/JSON to a CSV with columns:
       [question, answer, source, type]
       or download a processed CSV (e.g. from Kaggle or HF) and save as MedQuAD.csv.
    3. Put MedQuAD.csv in the same folder as this script.
    4. Run: streamlit run medical_chatbot_medquad.py
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st
MEDQUAD_CSV_PATH = "medquad.csv"  # path to processed MedQuAD CSV
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.45

@dataclass
class QAItem:
    question: str
    answer: str
    source: str
    focus_area: str


class MedQuADRetriever:
    def __init__(self, csv_path: str = MEDQUAD_CSV_PATH):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"MedQuAD CSV not found at {csv_path}. "
                f"Download/convert the dataset and save it as MedQuAD.csv."
            )

        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer(EMBED_MODEL)

        self.qa_items: List[QAItem] = []
        for _, row in self.df.iterrows():
            self.qa_items.append(
                QAItem(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    source=str(row.get("source", "")),
                    focus_area=str(row.get("focus_area", "")),
                )
            )

        questions = [item.question for item in self.qa_items]
        self.question_embeddings = self.model.encode(
            questions,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.question_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            item = self.qa_items[int(idx)]
            results.append(
                {
                    "question": item.question,
                    "answer": item.answer,
                    "score": float(score),
                    "source": item.source,
                    "focus_area": item.focus_area,
                }
            )
        return results


SYMPTOM_KEYWORDS = [
    "fever", "cough", "headache", "nausea", "fatigue", "pain", "vomiting",
    "diarrhea", "rash", "shortness of breath", "chest pain", "dizziness",
    "sore throat", "itching", "swelling"
]

DISEASE_KEYWORDS = [
    "diabetes", "hypertension", "asthma", "cancer", "covid", "flu",
    "heart disease", "migraine", "arthritis", "depression", "anxiety",
    "pneumonia", "tuberculosis"
]

TREATMENT_KEYWORDS = [
    "antibiotic", "insulin", "chemotherapy", "surgery", "therapy",
    "vaccination", "medication", "drug", "physiotherapy", "radiation",
    "tablet", "injection"
]


def simple_medical_ner(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    symptoms = [s for s in SYMPTOM_KEYWORDS if s in text_lower]
    diseases = [d for d in DISEASE_KEYWORDS if d in text_lower]
    treatments = [t for t in TREATMENT_KEYWORDS if t in text_lower]
    return {
        "symptoms": symptoms,
        "diseases": diseases,
        "treatments": treatments,
    }

@st.cache_resource
def load_retriever():
    return MedQuADRetriever(MEDQUAD_CSV_PATH)


retriever = load_retriever()

st.set_page_config(page_title="Medical Q&A Chatbot (MedQuAD)", page_icon="🩺")
st.title("Medical Q&A Chatbot 🩺")

st.write(
    "This chatbot uses the MedQuAD dataset for answering medical questions.\n\n"
    "**Important:** This is for educational purposes only and is **not** a "
    "substitute for professional medical advice, diagnosis, or treatment."
)

user_query = st.text_input(
    "Ask a medical question:",
    placeholder="Example: What are the symptoms of diabetes?"
)

if user_query:
    entities = simple_medical_ner(user_query)
    st.subheader("Detected medical entities")
    st.json(entities)

    st.subheader("Top answers from MedQuAD")
    results = retriever.search(user_query)

    any_shown = False
    for i, r in enumerate(results, start=1):
        if r["score"] < SIM_THRESHOLD:
            continue
        any_shown = True
        st.markdown(f"### Match {i} (similarity: {r['score']:.3f})")
        st.markdown(f"**Question:** {r['question']}")
        st.markdown(f"**Answer:** {r['answer']}")
        st.markdown(
            f"**Source:** {r['source']}  \n"
            f"**Focus Area:** {r['focus_area']}"
       )
        st.markdown("---")

    if not any_shown:
        st.write(
            "No sufficiently similar questions were found in MedQuAD. "
            "Please try rephrasing your question or consult a healthcare professional."
        )

    st.caption(
        "Always consult a qualified healthcare professional for medical advice. "
        "This tool may be incomplete or outdated."
    )