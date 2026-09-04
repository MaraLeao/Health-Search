# HealthSearch

Motor de busca híbrido para diretrizes médicas, combinando BM25 (léxico) com busca semântica vetorial (embeddings), unificados por Reciprocal Rank Fusion (RRF).

## O que o projeto faz

Resolve o problema de motores de busca puramente léxicos (que falham com sinônimos leigos, como "infarto" em vez de "síndrome coronariana aguda") e motores puramente semânticos (que perdem precisão em códigos exatos, como CÓD-ECG-12D, ou dosagens, como AAS 100mg).

## Arquivos

- `healthsearch_app.py`: aplicação Streamlit completa (interface, BM25, motor semântico e fusão RRF).
- `relatorio_tecnico_healthsearch.pdf`: relatório técnico descrevendo a arquitetura da solução.

## Como instalar

Recomenda-se usar um ambiente virtual (venv):

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install streamlit pandas numpy rank_bm25 scikit-learn
```

A biblioteca `sentence-transformers` é opcional. Se ela não estiver instalada, o app usa automaticamente um motor semântico simulado (documentado no código e no relatório), sem perda de funcionalidade. Para instalar a versão real:

```bash
pip install sentence-transformers
```

## Como rodar

```bash
streamlit run healthsearch_app.py
```

O terminal mostrará um endereço local (geralmente `http://localhost:8501`) que abre automaticamente no navegador.

## Como usar

1. Digite uma consulta clínica, por exemplo: `infarto`, `ataque cardiaco`, `CÓD-ECG-12D` ou `AAS 100mg`.
2. Ajuste os parâmetros na barra lateral:
   - `k1` e `b`: calibram o motor BM25.
   - `alfa`: define o peso entre BM25 e busca semântica na fusão RRF.
3. Veja os resultados nas abas: Híbrido (RRF), BM25, Semântico e Comparativo.

## Tecnologias

Python, Streamlit, rank_bm25, scikit-learn (TF-IDF) e, opcionalmente, sentence-transformers.

## Tela

<img width="1909" height="950" alt="image" src="https://github.com/user-attachments/assets/49d4f8d8-a90c-4a06-ba75-38d9854dedb1" />
