# 🎧 Spotify Express

Inteligência Artificial que recomenda músicas conforme o seu humor — protótipo educacional em **Streamlit + TensorFlow/Keras**.

Dois sliders (⚡ **Energia** e 🌧️ **Tristeza**, de 0 a 10) alimentam uma rede neural que classifica o seu momento em um de quatro gêneros: **Rock, Lofi, Pop ou Sertanejo** — com direito a um card estilo Spotify, probabilidades por gênero e uma rádio ao vivo para ouvir agora.

![gêneros](https://img.shields.io/badge/g%C3%AAneros-Rock%20%7C%20Lofi%20%7C%20Pop%20%7C%20Sertanejo-1DB954) ![feito com](https://img.shields.io/badge/feito%20com-Streamlit%20%2B%20TensorFlow-red)

---

## Como rodar (3 passos)

```bash
# 1. Instale as dependências
pip install streamlit tensorflow numpy

# 2. Rode o app
streamlit run app.py

# 3. Abra o endereço exibido no terminal
#    (padrão: http://localhost:8501)
```

### Para rodar os testes

```bash
pip install pytest
python -m pytest tests -v
```

---

## Como funciona

### 1. Dados sintéticos (`gerar_dados_sinteticos`)

Como é um protótipo educacional, o "comportamento do público" é simulado: cada gênero tem um centro de regra em (energia, tristeza) e os exemplos são amostrados com ruído gaussiano ao redor dele:

| Gênero | Energia | Tristeza | Lógica |
|---|---|---|---|
| 🎸 Rock | alta (≈7,5) | baixa (≈2,5) | cheio de energia, humor leve |
| 🤠 Sertanejo | alta (≈7,5) | alta (≈7,5) | agitado, com saudade no peito |
| 🌙 Lofi | baixa (≈2,5) | alta (≈7,5) | de boa, no mundo da lua |
| 🎤 Pop | baixa (≈2,5) | baixa (≈2,5) | tranquilo e sorridente |

São geradas ~4.000 amostras (1.000 por gênero) e o dataset é normalizado para `[0, 1]` (÷10).

### 2. Rede neural (`construir_modelo` / `treinar_modelo`)

Arquitetura simples e didática, implementada com `tf.keras`:

```
Entrada (2: energia + tristeza)
   ↓
Dense(12, ReLU)
   ↓
Dropout(0.2)          ← regularização
   ↓
Dense(4, Softmax)     ← probabilidade de cada gênero
```

- Perda `categorical_crossentropy` + otimizador `adam`
- Sementes fixas (`tf.keras.utils.set_random_seed`) → resultado **reproduzível**
- `@st.cache_resource` → a rede é treinada **uma única vez** por sessão; mover o slider não retreina

### 3. Predição e interface

A predição aplica a **mesma normalização do treino** (÷10). Na interface, um card verde `#1DB954` sobre fundo escuro `#121212` (estilo Spotify) mostra o gênero, uma frase de efeito e três músicas fictícias recomendadas.

### 4. Player de rádio (novidade)

O painel **💿 Player** reproduz estações **ao vivo e gratuitas** — rádios públicas do mundo inteiro indexadas pela API aberta **[radio-browser.info](https://www.radio-browser.info/)** (sem chave de API):

- Cada gênero busca estações pela tag correspondente (`rock`, `lofi`, `pop`, `sertanejo`), preferindo codecs MP3/AAC (compatíveis com o `<audio>` dos navegadores)
- **Rotação de espelhos**: se um servidor da API estiver fora do ar, tenta o próximo
- Cache de 1 hora (`@st.cache_data`) — mexer no slider não re-consulta a API
- **Catálogo embutido (fallback)**: se a rede bloquear a API (ex.: escola/proxy), o app usa automaticamente uma **lista pré-selecionada de streams públicos verificados** embutida em `app.py` — o player continua funcionando e avisa na interface ("🔌 ...")
- Fallback amigável em último caso: se não houver estação disponível, o app avisa em vez de quebrar

> ⚠️ **Requer internet.** As faixas dependem de streams públicos que podem sair do ar — é só trocar a estação no seletor do próprio app.

---

## Estrutura do projeto

```
proj2/
├── app.py                 # o app completo (único arquivo)
├── tests/
│   └── test_app.py        # 16 testes (pytest)
├── requirements.txt       # dependências de execução
├── README.md
└── docs/
    └── superpowers/       # spec e plano de implementação
        ├── specs/
        └── plans/
```

---

## Observações

- **Sem API do Spotify** de propósito: o escopo é didático e o player usa rádios públicas gratuitas.
- Em produção, troque a geração sintética por dados reais (ex.: features de áudio de uma API de música) — o pipeline da rede continuaria o mesmo.
- Este projeto foi construído com testes (TDD): a suíte valida formato/normalização dos dados, acurácia do modelo, mapeamento dos quadrantes, determinismo, o parsing das rádios e o fallback embutido — tudo sem depender da rede.

---

*Projeto educacional — Módulo de IA aplicada.*