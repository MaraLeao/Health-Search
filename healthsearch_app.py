# -*- coding: utf-8 -*-
"""
HealthSearch — Motor de Busca Híbrido (BM25 + Busca Semântica Vetorial + RRF)
==============================================================================
Desafio Integrador — UNIPÊ
Disciplina: Tendências em Ciência da Computação — Recuperação de Informação / PLN
Professor: Me. Ricardo Roberto de Lima

Como executar:
    pip install streamlit pandas numpy rank_bm25 scikit-learn sentence-transformers
    streamlit run healthsearch_app.py

Observação sobre o motor semântico:
    O app tenta carregar um modelo real de Sentence-Transformers
    ('paraphrase-multilingual-MiniLM-L12-v2'). Caso a biblioteca não esteja
    instalada ou não haja acesso à internet para baixar o modelo, o sistema
    cai automaticamente em um MODO DE SIMULAÇÃO VETORIAL, documentado na
    seção "Motor Semântico (Fase 3)" abaixo, que expande os termos da
    consulta/documentos para "conceitos clínicos" equivalentes (sinônimos
    médicos) e então vetoriza com TF-IDF + similaridade de cosseno. Isso
    reproduz, de forma didática, a capacidade de um motor semântico real de
    aproximar termos como "infarto" e "síndrome coronariana aguda".
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------
# Tenta habilitar embeddings reais (Sentence-Transformers). Se falhar por
# qualquer motivo (biblioteca ausente, sem internet para baixar o modelo,
# etc.), usamos o modo de simulação vetorial documentado na Fase 3.
# ------------------------------------------------------------------------
REAL_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util

    @st.cache_resource(show_spinner=False)
    def _load_sentence_model():
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    _model_test = _load_sentence_model()
    REAL_EMBEDDINGS_AVAILABLE = True
except Exception:
    REAL_EMBEDDINGS_AVAILABLE = False


# ==========================================================================
# FASE 1 — INGESTÃO DO CORPUS MÉDICO E PRÉ-PROCESSAMENTO
# ==========================================================================

CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "texto": (
            "Pacientes com dor precordial aguda e suspeita de síndrome "
            "coronariana devem realizar eletrocardiograma CÓD-ECG-12D em "
            "até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "texto": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes "
            "plaquetários reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "texto": (
            "A crise hipertensiva severa requer administração de "
            "anti-hipertensivos venosos e monitoramento contínuo da pressão "
            "arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "texto": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado "
            "com trombolíticos venosos em até quatro horas e meia do início "
            "dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "texto": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação precoce "
            "no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "texto": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca contínua por "
            "telemetria."
        ),
    },
]

# Stopwords em português (lista curada, evita dependência de downloads do NLTK)
STOPWORDS_PT = {
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "um", "uma",
    "uns", "umas", "para", "com", "sem", "por", "que", "se", "na", "no",
    "nas", "nos", "ao", "aos", "à", "às", "às", "os", "as", "ou", "mais",
    "menos", "muito", "pouco", "já", "também", "como", "quando", "onde",
    "qual", "quais", "este", "esta", "esse", "essa", "isso", "isto",
    "aquele", "aquela", "seu", "sua", "seus", "suas", "deve", "devem",
    "ser", "estar", "há", "num", "numa", "pelo", "pela", "pelos", "pelas",
    "até", "sob", "entre", "após", "sobre",
}


def preprocess(text: str, remove_stopwords: bool = True) -> list:
    """Fase 1: normalização, limpeza e tokenização do texto clínico."""
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ0-9\s\-]", " ", text)  # remove pontuação/caracteres especiais
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT]
    return tokens


# Pré-processa o corpus uma única vez
for doc in CORPUS:
    doc["tokens"] = preprocess(doc["texto"])

TOKENIZED_CORPUS = [doc["tokens"] for doc in CORPUS]
RAW_TEXTS = [doc["texto"] for doc in CORPUS]


# ==========================================================================
# FASE 3 — MOTOR SEMÂNTICO VETORIAL (com modo simulado documentado)
# ==========================================================================

# Dicionário de expansão de conceitos clínicos, usado apenas no MODO
# SIMULADO. Cada chave é um termo/sinônimo de entrada e o valor é o
# "conceito canônico" ao qual ele pertence — assim, termos leigos e termos
# técnicos passam a compartilhar dimensões no espaço vetorial (TF-IDF),
# imitando o comportamento de um embedding semântico real.
CONCEITOS_CLINICOS = {
    "infarto": "conceito_sindrome_coronariana_aguda",
    "ataque": "conceito_sindrome_coronariana_aguda",
    "cardiaco": "conceito_sindrome_coronariana_aguda",
    "coronariana": "conceito_sindrome_coronariana_aguda",
    "isquemia": "conceito_sindrome_coronariana_aguda",
    "miocardica": "conceito_sindrome_coronariana_aguda",
    "miocardio": "conceito_sindrome_coronariana_aguda",
    "precordial": "conceito_sindrome_coronariana_aguda",
    "aas": "conceito_antiagregante_plaquetario",
    "acetilsalicilico": "conceito_antiagregante_plaquetario",
    "antiagregantes": "conceito_antiagregante_plaquetario",
    "plaquetarios": "conceito_antiagregante_plaquetario",
    "avc": "conceito_acidente_vascular_cerebral",
    "derrame": "conceito_acidente_vascular_cerebral",
    "vascular": "conceito_acidente_vascular_cerebral",
    "cerebral": "conceito_acidente_vascular_cerebral",
    "isquemico": "conceito_acidente_vascular_cerebral",
    "tromboliticos": "conceito_acidente_vascular_cerebral",
    "parada": "conceito_parada_cardiorrespiratoria",
    "cardiorrespiratoria": "conceito_parada_cardiorrespiratoria",
    "reanimacao": "conceito_parada_cardiorrespiratoria",
    "rcr": "conceito_parada_cardiorrespiratoria",
    "desfibrilacao": "conceito_parada_cardiorrespiratoria",
    "hipertensao": "conceito_crise_hipertensiva",
    "hipertensiva": "conceito_crise_hipertensiva",
    "pressao": "conceito_crise_hipertensiva",
    "hipertensivos": "conceito_crise_hipertensiva",
    "ecg": "conceito_eletrocardiograma",
    "eletrocardiograma": "conceito_eletrocardiograma",
    "arritmias": "conceito_eletrocardiograma",
    "telemetria": "conceito_eletrocardiograma",
}


def expandir_conceitos(tokens: list) -> str:
    """Expande tokens com seus conceitos clínicos equivalentes (modo simulado)."""
    expandido = list(tokens)
    for tok in tokens:
        tok_norm = re.sub(r"[^a-z0-9]", "", tok)
        if tok_norm in CONCEITOS_CLINICOS:
            expandido.append(CONCEITOS_CLINICOS[tok_norm])
    return " ".join(expandido)


@st.cache_resource(show_spinner=False)
def _build_simulated_vectorizer(corpus_expandido):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus_expandido)
    return vectorizer, matrix


def semantic_search(query: str):
    """Fase 3: retorna array de similaridade de cosseno (consulta x docs)."""
    if REAL_EMBEDDINGS_AVAILABLE:
        model = _load_sentence_model()
        doc_embeddings = model.encode(RAW_TEXTS, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        sims = st_util.cos_sim(query_embedding, doc_embeddings).cpu().numpy().flatten()
        return sims
    else:
        # Modo simulado: expande conceitos e usa TF-IDF + cosseno
        corpus_expandido = [expandir_conceitos(doc["tokens"]) for doc in CORPUS]
        query_tokens = preprocess(query)
        query_expandida = expandir_conceitos(query_tokens)
        vectorizer, doc_matrix = _build_simulated_vectorizer(tuple(corpus_expandido))
        query_vec = vectorizer.transform([query_expandida])
        sims = cosine_similarity(query_vec, doc_matrix).flatten()
        return sims


# ==========================================================================
# FASE 2 — MOTOR LÉXICO (BM25) COM PARÂMETROS INTERATIVOS
# ==========================================================================

def bm25_search(query: str, k1: float, b: float):
    """Fase 2: retorna scores do BM25 Okapi calibrado por k1 e b."""
    bm25 = BM25Okapi(TOKENIZED_CORPUS, k1=k1, b=b)
    query_tokens = preprocess(query)
    scores = bm25.get_scores(query_tokens)
    return scores


# ==========================================================================
# FASE 4 — FUSÃO RRF (RECIPROCAL RANK FUSION)
# ==========================================================================

K_RRF = 60  # constante de suavização de posição


def scores_to_ranks(scores: np.ndarray) -> dict:
    """Converte um vetor de scores em um dicionário {indice_doc: rank (1..N)}."""
    ordem = np.argsort(-scores)  # ordem decrescente de score
    ranks = {}
    for posicao, idx in enumerate(ordem):
        ranks[idx] = posicao + 1
    return ranks


def rrf_fusion(rank_bm25: dict, rank_semantico: dict, alpha: float, k_rrf: int = K_RRF) -> dict:
    """
    Fase 4: Score_RRF(D) = alpha * [1/(k_rrf + Rank_BM25)] +
                           (1 - alpha) * [1/(k_rrf + Rank_Semantico)]
    """
    score_rrf = {}
    for idx in range(len(CORPUS)):
        r_bm25 = rank_bm25[idx]
        r_sem = rank_semantico[idx]
        score = alpha * (1.0 / (k_rrf + r_bm25)) + (1.0 - alpha) * (1.0 / (k_rrf + r_sem))
        score_rrf[idx] = score
    return score_rrf


# ==========================================================================
# INTERFACE STREAMLIT
# ==========================================================================

st.set_page_config(page_title="HealthSearch — Busca Híbrida", page_icon="🩺", layout="wide")

st.title("🩺 HealthSearch — Motor de Busca Híbrido")
st.caption(
    "BM25 (léxico) + Busca Semântica Vetorial (embeddings) + Reciprocal Rank Fusion (RRF) "
    "— Desafio Integrador UNIPÊ"
)

if REAL_EMBEDDINGS_AVAILABLE:
    st.success("Motor semântico: embeddings reais (Sentence-Transformers) carregados.", icon="✅")
else:
    st.info(
        "Motor semântico rodando em **modo de simulação vetorial documentado** "
        "(expansão de conceitos clínicos + TF-IDF + similaridade de cosseno), pois a "
        "biblioteca `sentence-transformers` não pôde ser carregada neste ambiente "
        "(ausência da lib ou do modelo baixado). O comportamento de aproximar sinônimos "
        "médicos é preservado — veja a seção 'Motor Semântico' no código.",
        icon="ℹ️",
    )

# ---------------------------- Sidebar -----------------------------------
st.sidebar.header("⚙️ Calibração do Motor")

st.sidebar.subheader("Fase 2 — BM25 (Léxico)")
k1 = st.sidebar.slider("k1 — Saturação de Frequência", 0.0, 3.0, 1.2, 0.1)
b = st.sidebar.slider("b — Normalização por Tamanho", 0.0, 1.0, 0.75, 0.05)

st.sidebar.subheader("Fase 4 — Fusão RRF")
alpha = st.sidebar.slider(
    "α — Peso BM25 vs Semântico", 0.0, 1.0, 0.5, 0.05,
    help="α = 1.0 → 100% léxico (BM25) | α = 0.0 → 100% semântico"
)
st.sidebar.caption(f"k_RRF (constante fixa) = {K_RRF}")

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Consultas de Exemplo")
exemplos = {
    "Sinônimo leigo → termo técnico": "infarto",
    "Termo leigo → 'ataque cardíaco'": "ataque cardiaco",
    "Código exato de exame": "CÓD-ECG-12D",
    "Dosagem exata de medicamento": "AAS 100mg",
}
exemplo_escolhido = st.sidebar.selectbox("Carregar exemplo:", ["(nenhum)"] + list(exemplos.keys()))

# ---------------------------- Query input --------------------------------
default_query = exemplos.get(exemplo_escolhido, "") if exemplo_escolhido != "(nenhum)" else ""
query = st.text_input("Digite sua consulta clínica:", value=default_query, placeholder="ex.: infarto, CÓD-ECG-12D, AAS 100mg...")

buscar = st.button("🔍 Buscar", type="primary")

if buscar or query:
    if not query.strip():
        st.warning("Digite uma consulta para buscar.")
    else:
        # --- Fase 1: tokens da consulta ---
        query_tokens = preprocess(query)
        with st.expander("🧪 Fase 1 — Pré-processamento da consulta"):
            st.write(f"**Consulta original:** {query}")
            st.write(f"**Tokens após limpeza/stopwords:** {query_tokens}")

        # --- Fase 2: BM25 ---
        bm25_scores = bm25_search(query, k1, b)
        rank_bm25 = scores_to_ranks(bm25_scores)

        # --- Fase 3: Semântico ---
        sem_scores = semantic_search(query)
        rank_semantico = scores_to_ranks(sem_scores)

        # --- Fase 4: RRF ---
        rrf_scores = rrf_fusion(rank_bm25, rank_semantico, alpha)

        # Monta DataFrame consolidado
        rows = []
        for idx, doc in enumerate(CORPUS):
            rows.append({
                "ID": doc["id"],
                "Título": doc["titulo"],
                "Trecho": doc["texto"],
                "Score BM25": round(float(bm25_scores[idx]), 4),
                "Rank BM25": rank_bm25[idx],
                "Score Semântico": round(float(sem_scores[idx]), 4),
                "Rank Semântico": rank_semantico[idx],
                "Score RRF": round(float(rrf_scores[idx]), 5),
            })
        df = pd.DataFrame(rows)
        df_rrf_sorted = df.sort_values("Score RRF", ascending=False).reset_index(drop=True)
        df_rrf_sorted.insert(0, "Rank Final", range(1, len(df_rrf_sorted) + 1))

        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Híbrido (RRF)", "🔤 BM25 (Léxico)", "🧠 Semântico (Vetorial)", "📊 Comparativo & Métricas"
        ])

        with tab1:
            st.subheader("Ranking Final — Reciprocal Rank Fusion")
            st.dataframe(
                df_rrf_sorted[["Rank Final", "ID", "Título", "Trecho", "Score RRF", "Rank BM25", "Rank Semântico"]],
                width='stretch', hide_index=True,
            )
            st.markdown(f"**Documento mais relevante:** `{df_rrf_sorted.iloc[0]['ID']}` — {df_rrf_sorted.iloc[0]['Título']}")

        with tab2:
            st.subheader("Ranking Léxico — BM25 Okapi")
            df_bm25_sorted = df.sort_values("Rank BM25").reset_index(drop=True)
            st.dataframe(
                df_bm25_sorted[["Rank BM25", "ID", "Título", "Trecho", "Score BM25"]],
                width='stretch', hide_index=True,
            )
            st.caption(f"Parâmetros atuais: k1 = {k1} | b = {b}")

        with tab3:
            st.subheader("Ranking Semântico — Similaridade de Cosseno")
            df_sem_sorted = df.sort_values("Rank Semântico").reset_index(drop=True)
            st.dataframe(
                df_sem_sorted[["Rank Semântico", "ID", "Título", "Trecho", "Score Semântico"]],
                width='stretch', hide_index=True,
            )
            modo = "Embeddings reais (Sentence-Transformers)" if REAL_EMBEDDINGS_AVAILABLE else "Simulação vetorial (expansão de conceitos + TF-IDF)"
            st.caption(f"Modo do motor semântico: {modo}")

        with tab4:
            st.subheader("Comparativo de Ranks entre Motores")
            chart_df = df.set_index("ID")[["Rank BM25", "Rank Semântico"]]
            st.bar_chart(chart_df)
            st.caption("Ranks menores (barras mais baixas) indicam maior relevância. Compare onde BM25 e Semântico divergem.")

            st.subheader("Métricas de Desempenho da Busca")
            c1, c2, c3 = st.columns(3)
            c1.metric("Top-1 BM25", df.loc[df["Rank BM25"] == 1, "ID"].values[0])
            c2.metric("Top-1 Semântico", df.loc[df["Rank Semântico"] == 1, "ID"].values[0])
            c3.metric("Top-1 Híbrido (RRF)", df_rrf_sorted.iloc[0]["ID"])

            concordancia = sum(1 for idx in range(len(CORPUS)) if rank_bm25[idx] == rank_semantico[idx])
            st.caption(f"Documentos com o mesmo rank em ambos os motores: {concordancia}/{len(CORPUS)}")

st.markdown("---")
with st.expander("📚 Base de Documentos (Corpus Médico Completo)"):
    st.dataframe(pd.DataFrame(CORPUS)[["id", "titulo", "texto"]], width='stretch', hide_index=True)
