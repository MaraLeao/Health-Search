# HealthSearch

Um **motor de busca híbrido** desenvolvido com tecnologias modernas de Recuperação de Informação e Processamento de Linguagem Natural (PLN).

## Sobre o Projeto

HealthSearch combina três estratégias de busca para entregar resultados precisos e semanticamente relevantes em documentos médicos e científicos:

| Estratégia | Descrição |
|-----------|-----------|
| **BM25** | Algoritmo probabilístico de ranking, especializado em compatibilidade de termos exatos |
| **Busca Semântica Vetorial** | Embeddings baseados em deep learning (Sentence-Transformers) para capturar significado contextual |
| **RRF (Reciprocal Rank Fusion)** | Combina rankings de múltiplos modelos para melhor precisão |

## Características

- **Interface Intuitiva** - Desenvolvida com [Streamlit](https://streamlit.io)  
- **Modelo Semântico** - Suporte a `paraphrase-multilingual-MiniLM-L12-v2` com fallback para modo simulado  
- **Busca Avançada** - Combina BM25 + Embeddings + RRF  
- **Multilíngue** - Suporte a múltiplos idiomas através do modelo de transformers  
- **Modo de Simulação** - Funciona mesmo sem conexão com internet (expande termos em conceitos médicos equivalentes)  

## Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Setup do Ambiente

```bash
# Clone o repositório
git clone <seu-repositorio>
cd Health-Search

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

## Dependências

- **streamlit** - Framework para web apps em Python
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **rank-bm25** - Implementação do algoritmo BM25
- **scikit-learn** - Machine learning (TF-IDF, similaridade cosseno)
- **sentence-transformers** - Embeddings de alta qualidade (opcional, com fallback)

## Como Usar

```bash
streamlit run healthsearch_app.py
```

A aplicação abrirá em seu navegador padrão (geralmente `http://localhost:8501`).

### Exemplo de Uso

1. Insira sua **consulta** na caixa de busca
2. Ajuste o **número de resultados** desejados
3. (Opcional) Configure pesos para BM25 vs Busca Semântica
4. Visualize os resultados com scores de relevância

## Arquitetura

### Motor BM25 (Fase 1)
- Tokenização e pré-processamento de documentos
- Cálculo de scores BM25 para cada documento
- Ranking por relevância

### Motor Semântico (Fase 2 e 3)
- **Modo Real**: Usa embeddings do Sentence-Transformers
- **Modo Simulado**: Expande termos em conceitos clínicos (ex: "infarto" → "síndrome coronariana aguda") e vetoriza com TF-IDF + similaridade cosseno

### Fusão de Rankings (Fase 4)
- RRF (Reciprocal Rank Fusion) combina scores dos modelos
- Resultado final ordenado por score agregado

## Estrutura do Projeto

```
Health-Search/
├── healthsearch_app.py          # Aplicação principal
├── HealthSearch/
│   ├── healthsearch_app.py      # Módulo auxiliar
│   └── ...
├── requirements.txt              # Dependências Python
├── .gitignore                    # Padrões de ignore do Git
├── README.md                     # Este arquivo
└── relatorio_tecnico_healthsearch.pdf  # Documentação técnica
```

## Contexto Acadêmico

**Instituição:** UNIPÊ  
**Disciplina:** Tendências em Ciência da Computação — Recuperação de Informação / PLN  
**Professor:** Me. Ricardo Roberto de Lima  
**Tipo:** Desafio Integrador  

## Notas Importantes

**Sobre Embeddings:**  
Se o Sentence-Transformers não estiver instalado ou houver erro ao baixar o modelo pré-treinado, o sistema cai automaticamente em **modo de simulação vetorial**. Isso garante que a aplicação sempre funcione, com graceful degradation.

**Dica de Performance:**  
Para trabalhar com grandes corpora de documentos, considere:
- Usar a versão menor do modelo de transformers
- Implementar caching de embeddings
- Aumentar recursos de memória

## Contribuindo

Sugestões e melhorias são bem-vindas! Por favor:
1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -am 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

