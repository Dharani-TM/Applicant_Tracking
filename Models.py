import gensim
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import streamlit as st


# ---------------- Mean Pooling ---------------- #

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


# ---------------- HuggingFace BERT ---------------- #

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/bert-base-nli-mean-tokens"
    )

    model = AutoModel.from_pretrained(
        "sentence-transformers/bert-base-nli-mean-tokens"
    )

    return tokenizer, model


@st.cache_data
def get_HF_embeddings(sentence):

    tokenizer, model = load_model()

    encoded_input = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )

    return embeddings


# ---------------- Doc2Vec ---------------- #

@st.cache_data
def get_doc2vec_embeddings(JD, text_resume):

    nltk.download("punkt", quiet=True)

    tagged_data = [
        TaggedDocument(
            words=word_tokenize(JD.lower()),
            tags=["0"]
        )
    ]

    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=512,
        min_count=3,
        epochs=80
    )

    model.build_vocab(tagged_data)

    model.train(
        tagged_data,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    JD_embeddings = np.transpose(
        model.dv["0"].reshape(-1, 1)
    )

    resume_embeddings = []

    for resume in text_resume:

        text = word_tokenize(resume.lower())

        embedding = model.infer_vector(text)

        resume_embeddings.append(
            np.transpose(
                embedding.reshape(-1, 1)
            )
        )

    return JD_embeddings, resume_embeddings


# ---------------- Cosine Similarity ---------------- #

def cosine(embeddings1, embeddings2):

    scores = []

    for emb in embeddings1:

        similarity = cosine_similarity(
            np.array(emb),
            np.array(embeddings2)
        )

        score = round(float(similarity[0][0]) * 100, 2)

        scores.append(score)

    return scores
