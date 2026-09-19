# ============================================================================
#  Spotify Express — recomendação musical com redes neurais
#  Seções: 1) Imports  2) Constantes/catálogo  3) Dados sintéticos
#          4) Modelo TensorFlow  5) Interface Streamlit
#  (As seções 4 e 5 são adicionadas nas Tarefas 2 e 3)
# ============================================================================

import numpy as np
import requests
import streamlit as st
import tensorflow as tf

# --- 2) Constantes globais e catálogo dos gêneros ---------------------------

# Ordem oficial das 4 classes (o índice é a posição no one-hot)
CLASSES = ["Rock", "Lofi", "Pop", "Sertanejo"]

# Centros das regras de humor: (energia, tristeza) na escala 0-10
REGRA_CENTROS = {
    "Rock":      (7.5, 2.5),   # alta energia + tristeza baixa
    "Sertanejo": (7.5, 7.5),   # alta energia + tristeza alta
    "Lofi":      (2.5, 7.5),   # baixa energia + tristeza alta
    "Pop":       (2.5, 2.5),   # baixa energia + tristeza baixa
}

# Catálogo visual de cada gênero (capa, frase e 3 músicas fictícias)
GENRES_INFO = {
    "Rock": {
        "emoji": "🎸",
        "frase": "Alta energia e coração em chamas: hora de soltar o arrepio.",
        "musicas": ["Asfalto e Tempestade", "Fogo no Volante", "Corrente Elétrica"],
    },
    "Sertanejo": {
        "emoji": "🤠",
        "frase": "Pé na estrada, viola no peito e saudade no ar.",
        "musicas": ["Nota da Moda", "Botina e Lua", "Caminhão e Coração"],
    },
    "Lofi": {
        "emoji": "🌙",
        "frase": "Respira fundo... o mundo pode esperar mais um minuto.",
        "musicas": ["Chuva na Janela", "Madrugada em Slow Motion", "Café às 3h"],
    },
    "Pop": {
        "emoji": "🎤",
        "frase": "Vibes leves e sorriso no rosto: bora dançar.",
        "musicas": ["Neon no Coração", "Coração em Marte", "Doce Euforia"],
    },
}

# --- 3) Geração dos dados sintéticos de treino -------------------------------

def gerar_dados_sinteticos(n_por_classe=1000, seed=42):
    """Gera o dataset sintético seguindo as regras de humor + ruído gaussiano.

    Por que dados sintéticos? É um protótipo educacional: as regras são
    conhecidas, então geramos exemplos ao redor delas para a rede aprender
    o padrão coerente antes da predição.

    Retorna:
        X:        (n_amostras, 2) energia e tristeza NORMALIZADAS em [0, 1]
        y_onehot: (n_amostras, 4) labels one-hot
        y:        (n_amostras,) índices das classes (0..3)
    """
    rng = np.random.default_rng(seed)
    lista_x, lista_y = [], []

    for indice, genero in enumerate(CLASSES):
        c_energia, c_tristeza = REGRA_CENTROS[genero]   # centro da regra

        # Amostras ao redor do centro com ruído gaussiano (variação realista)
        energia = rng.normal(c_energia, 1.2, n_por_classe)
        tristeza = rng.normal(c_tristeza, 1.2, n_por_classe)

        # Limita aos limites físicos da escala (0-10)
        energia = np.clip(energia, 0.0, 10.0)
        tristeza = np.clip(tristeza, 0.0, 10.0)

        lista_x.append(np.column_stack([energia, tristeza]))
        lista_y.append(np.full(n_por_classe, indice, dtype=np.int32))

    X = np.vstack(lista_x).astype(np.float32)
    y = np.concatenate(lista_y)

    # Normalização para [0, 1]: ajuda a rede a convergir e é o MESMO passo
    # que aplicaremos na predição (consistência treino/predição).
    X = X / 10.0

    # Converte rótulos para one-hot (4 colunas: Rock, Lofi, Pop, Sertanejo)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

    # Embaralha o dataset de forma determinística. Por que? Sem embaralhar,
    # as amostras ficam agrupadas por classe e o validation_split do fit
    # (últimos 20%) pegaria só um gênero, desbalanceando o treino.
    perm = rng.permutation(len(y))
    X, y, y_onehot = X[perm], y[perm], y_onehot[perm]

    return X, y_onehot, y

# --- 4) Modelo TensorFlow/Keras ----------------------------------------------

def construir_modelo():
    """Monta a arquitetura da rede neural.

    2 entradas (energia + tristeza) -> camada oculta ReLU -> softmax 4 saídas.
    A saída softmax devolve a probabilidade de cada um dos 4 gêneros.
    """
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),                # entrada explícita (idioma atual do Keras)
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dropout(0.2),   # regularização: evita overfitting
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",   # perda padrão p/ classificação multiclasse
        metrics=["accuracy"],
    )
    return modelo


def treinar_modelo(X, y_onehot, epochs=60, seed=42):
    """Treina o modelo com sementes fixas (resultado reprodutível).

    Usamos tf.keras.utils.set_random_seed, que re-semeia Python, NumPy,
    TensorFlow E Keras num único passo. É essencial no Keras 3: apenas
    np.random.seed/tf.random.set_seed NÃO controlam a inicialização dos
    pesos do Keras, então dois treinos com a mesma semente dariam modelos
    diferentes (e recomendações diferentes ao reabrir o app).
    """
    tf.keras.utils.set_random_seed(seed)

    modelo = construir_modelo()
    modelo.fit(
        X, y_onehot,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,   # 20% reservados para validar durante o treino
        verbose=0,              # silencia o progresso no terminal
    )
    return modelo


@st.cache_resource(show_spinner="🎧 Treinando a rede neural (só na primeira vez)...")
def carregar_modelo():
    """Gera os dados e treina a rede UMA vez por sessão.

    O decorator @st.cache_resource faz a mágica do protótipo: ao mexer no
    slider (o Streamlit re-executa o script inteiro), o modelo já treinado
    é devolvido do cache em vez de ser treinado de novo.
    """
    X, y_onehot, _ = gerar_dados_sinteticos()
    return treinar_modelo(X, y_onehot)


# --- Predição: mesma normalização usada no treino -----------------------------

def prever_genero(energia, tristeza, modelo):
    """Classifica o humor (energia, tristeza em 0-10) em um dos 4 gêneros.

    Retorna:
        probs:      dict {gênero: probabilidade}
        genero_top: nome do gênero com maior probabilidade
    """
    entrada = np.array([[energia / 10.0, tristeza / 10.0]], dtype=np.float32)
    probabilidades = modelo.predict(entrada, verbose=0)[0]

    genero_top = CLASSES[int(np.argmax(probabilidades))]
    probs = {g: float(p) for g, p in zip(CLASSES, probabilidades)}
    return probs, genero_top

# --- 5) Interface Streamlit (visual estilo Spotify) ---------------------------

# CSS customizado: fundo escuro #121212 e verde #1DB954 (identidade Spotify)
CSS_SPOTIFY = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #181818;
    }
    h1, h2, h3, h4, p, label, [data-testid="stCaptionContainer"] {
        color: #FFFFFF;
    }
    /* Card do gênero recomendado */
    .spotify-card {
        background: #181818;
        border: 2px solid #1DB954;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55);
        color: #FFFFFF;
    }
    .spotify-cover {
        font-size: 4.5rem;
        line-height: 1.2;
    }
    .spotify-genre {
        color: #1DB954;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .spotify-tagline {
        color: #B3B3B3;
        font-size: 1.05rem;
        font-style: italic;
        margin-bottom: 0.9rem;
    }
    /* Lista de músicas */
    .musica-item {
        background: #242424;
        border-left: 4px solid #1DB954;
        border-radius: 6px;
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.45rem;
        color: #FFFFFF;
    }
    .musica-num {
        color: #1DB954;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    /* Rótulos das probabilidades */
    .prob-label {
        color: #B3B3B3;
    }
</style>
"""


def rotulo_qualitativo(valor):
    """Traduz um valor 0-10 para baixa/média/alta (textos da UI)."""
    if valor <= 3:
        return "baixa"
    if valor <= 7:
        return "média"
    return "alta"


# --- 6) Player: rádios públicas e gratuitas (radio-browser.info) --------------
# A API aberta radio-browser.info lista milhares de rádios públicas do mundo
# inteiro — gratuita e sem chave de API. Cada gênero é buscado pela tag.

RADIO_TAGS = {"Rock": "rock", "Lofi": "lofi", "Pop": "pop", "Sertanejo": "sertanejo"}

# A API tem vários espelhos; "all" redireciona para um disponível. Se um
# estiver fora do ar, tentamos o próximo (rotação = resiliência).
RADIO_MIRRORS = [
    "https://all.api.radio-browser.info/json/stations/search",
    "https://de1.api.radio-browser.info/json/stations/search",
    "https://de2.api.radio-browser.info/json/stations/search",
    "https://nl1.api.radio-browser.info/json/stations/search",
    "https://at1.api.radio-browser.info/json/stations/search",
]

# Codecs com melhor compatibilidade com o <audio> dos navegadores
CODECS_PREFERIDOS = ("MP3", "AAC")
FORMATO_AUDIO = {"MP3": "audio/mpeg", "AAC": "audio/aac", "OGG": "audio/ogg"}

# Catálogo embutido: streams públicos e gratuitos verificados ao vivo.
# Quando a rede filtra/bloqueia a API (ex.: escola, proxy), o player usa esta
# lista — assim o app NUNCA fica sem rádio só por causa de um bloqueio.
# Todas as URLs são HTTPS e codecs MP3/AAC para tocar em qualquer navegador.
# (Verificadas em 19/09/2026.)
_CATALOGO_EMBUTIDO = {
    "Rock": [
        ("SomaFM Indie Pop Rocks", "https://ice2.somafm.com/indiepop-128-aac", "AAC"),
        ("NME Radio", "https://listen-msmn.sharp-stream.com/nme1.mp3", "MP3"),
    ],
    "Lofi": [
        ("REYFM Lofi", "https://listen.reyfm.de/lofi_320kbps.mp3", "MP3"),
        ("NIA Radio Lo-Fi", "https://radio.nia.nc/radio/8020/lofi-hq-stream.aac", "AAC"),
    ],
    "Pop": [
        ("EuroDance 90 Radio", "https://stream-eurodance90.fr/radio/8000/128.mp3", "MP3"),
        ("Kiss FM Romênia", "https://live.kissfm.ro/kissfm.aacp", "AAC"),
    ],
    "Sertanejo": [
        ("Rádio Buteco Sertanejo", "https://stream.zeno.fm/6kumndewqbruv", "AAC"),
        ("Hunter FM Hits Brasil", "https://live.hunter.fm/hitsbrasil_normal", "AAC"),
    ],
}

# Mesmo formato de dicionário usado pelas estações vindas da API
ESTACOES_PADRAO = {
    genero: [
        {"nome": nome, "url": url, "codec": codec, "formato": FORMATO_AUDIO.get(codec)}
        for nome, url, codec in _CATALOGO_EMBUTIDO[genero]
    ]
    for genero in CLASSES
}


def filtrar_estacoes(estacoes_json, limite=5):
    """Seleciona as melhores estações a partir da resposta JSON da API.

    A API já vem ordenada por votos; aqui removemos URLs vazias/duplicadas e
    priorizamos codecs MP3/AAC (tocam direto no navegador). O formato MIME é
    guardado para o st.audio saber como decodificar o stream.
    """
    vistas, resultado = set(), []
    for estacao in estacoes_json:
        url = (estacao.get("url_resolved") or "").strip()
        codec = (estacao.get("codec") or "").upper()
        if not url or url in vistas:
            continue
        vistas.add(url)
        resultado.append({
            "nome": (estacao.get("name") or "Estação sem nome").strip(),
            "url": url,
            "codec": codec,
            "formato": FORMATO_AUDIO.get(codec),
        })
    # MP3/AAC primeiro (melhor compatibilidade); mantém a ordem por votos
    # dentro de cada grupo, pois a lista já veio ordenada da API.
    resultado.sort(key=lambda e: 0 if e["codec"] in CODECS_PREFERIDOS else 1)
    return resultado[:limite]


def estacoes_sao_embutidas(genero, estacoes):
    """True quando a lista veio do catálogo embutido (API bloqueada/indisponível)."""
    urls_padrao = [e["url"] for e in ESTACOES_PADRAO.get(genero, [])]
    urls_atual = [e["url"] for e in estacoes]
    return bool(urls_padrao) and urls_atual == urls_padrao


def buscar_estacoes(genero, limite=5, timeout=10):
    """Consulta a API por estações do gênero, com fallback silencioso.

    Tenta cada espelho da API em ordem; a primeira resposta válida vence.
    Se TODOS os espelhos falharem (rede bloqueada, ex.: escola/proxy),
    devolve o catálogo embutido `ESTACOES_PADRAO` — assim o player continua
    funcionando mesmo sem acesso à API.
    """
    parametros = {
        "tag": RADIO_TAGS[genero],
        "limit": 50,
        "hidebroken": "true",   # só estações checadas recentemente
        "order": "votes",
        "reverse": "true",
        "fields": "name,url_resolved,codec",
    }
    cabecalhos = {"User-Agent": "SpotifyExpress/1.0 (protótipo educacional)"}
    for url in RADIO_MIRRORS:
        try:
            resposta = requests.get(url, params=parametros, timeout=timeout, headers=cabecalhos)
            if resposta.status_code != 200:
                continue   # espelho indisponível, tenta o próximo
            return filtrar_estacoes(resposta.json(), limite=limite)
        except Exception:
            continue
    # Fallback: avisa no terminal (diagnóstico) e usa a lista embutida
    print(
        "[Player] API de rádios inacessível (rede filtrada/proxy?) — usando "
        f"a lista embutida de {len(ESTACOES_PADRAO.get(genero, []))} estações "
        f"para {genero}."
    )
    return list(ESTACOES_PADRAO.get(genero, []))


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_estacoes(genero):
    """Estações do gênero com cache de 1 h.

    O @st.cache_data evita consultar a rádio a CADA rerun: mover o slider
    (que re-executa o script) não precisa re-buscar a lista de estações.
    """
    return buscar_estacoes(genero)


def main():
    # st.set_page_config PRECISA ser o primeiro comando do Streamlit
    st.set_page_config(page_title="Spotify Express", page_icon="🎧", layout="wide")
    st.markdown(CSS_SPOTIFY, unsafe_allow_html=True)

    # ----- Cabeçalho -----
    st.title("🎧 **Spotify Express**")
    st.caption("Inteligência Artificial que recomenda músicas conforme o seu humor.")

    # ----- Sidebar: controles do usuário -----
    with st.sidebar:
        st.markdown("### 🙋 Como você está?")
        st.caption("Ajuste os sliders para descrever seu momento.")

        energia = st.slider(
            "⚡ Energia", 0, 10, 5,
            help="0 = sem energia nenhuma · 10 = cheio(a) de energia",
        )
        tristeza = st.slider(
            "🌧️ Tristeza", 0, 10, 5,
            help="0 = tranquilo(a) · 10 = muito triste",
        )

        st.divider()
        st.info(
            "💡 Uma rede neural (TensorFlow/Keras) foi treinada com dados "
            "sintéticos para aprender o padrão humor → gênero."
        )
        st.caption(
            f"Seu momento: **energia {rotulo_qualitativo(energia)}** e "
            f"**tristeza {rotulo_qualitativo(tristeza)}**."
        )

    # ----- Predição (modelo vem do cache, não retreina ao mover o slider) -----
    modelo = carregar_modelo()
    probs, genero = prever_genero(energia, tristeza, modelo)
    info = GENRES_INFO[genero]

    # ----- Layout principal estilo Spotify (lado a lado com st.columns) -----
    col_esquerda, col_direita = st.columns([3, 2], gap="large")

    with col_esquerda:
        itens_musicas = "".join(
            f'<div class="musica-item"><span class="musica-num">{i}.</span>'
            f"{nome}</div>"
            for i, nome in enumerate(info["musicas"], start=1)
        )
        st.markdown(
            f"""
            <div class="spotify-card">
                <div class="spotify-cover">{info['emoji']}</div>
                <div class="spotify-genre">{genero.upper()}</div>
                <div class="spotify-tagline">{info['frase']}</div>
                <div style="margin-top: 0.4rem;"><b>🎵 Recomendadas para você:</b></div>
                {itens_musicas}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_direita:
        st.markdown("### 🧠 Confiança do modelo")
        st.caption("Probabilidade que a rede neural atribuiu a cada gênero:")
        for genero_item in CLASSES:
            probabilidade = probs[genero_item]
            destaque = " ✅" if genero_item == genero else ""
            st.markdown(
                f'<div class="prob-label"><b>{genero_item}</b>{destaque}</div>',
                unsafe_allow_html=True,
            )
            st.progress(probabilidade, text=f"{probabilidade * 100:.1f}%")

        st.divider()
        st.markdown("### 💿 Player")
        estacoes = carregar_estacoes(genero)
        if not estacoes:
            st.info(
                "📡 Nenhuma rádio disponível no momento para este gênero. "
                "Verifique a internet e mexa em um slider para tentar de novo."
            )
        else:
            usando_embutidas = estacoes_sao_embutidas(genero, estacoes)
            rotulos = {f"{e['nome']} ({e['codec']})": e for e in estacoes}
            selecao = st.selectbox("Estação ao vivo", list(rotulos), key="estacao_player")
            estacao = rotulos[selecao]
            if estacao["formato"]:
                st.audio(estacao["url"], format=estacao["formato"])
            else:
                st.audio(estacao["url"])
            if usando_embutidas:
                st.caption(
                    "🔌 Sua rede bloqueou a API de rádios — usando a lista "
                    "pré-selecionada embutida no app (streams públicos e "
                    "gratuitos). Se a estação cair, escolha outra acima."
                )
            else:
                st.caption(
                    "📻 Rádios públicas e gratuitas (radio-browser.info) — "
                    "requer internet. Se a estação cair, escolha outra acima."
                )


# Garante que importar o app (para testes) NÃO abra a interface
if __name__ == "__main__":
    main()