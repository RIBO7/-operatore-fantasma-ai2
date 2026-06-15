import streamlit as st
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("👻 Operatore Fantasma AI")

if "testi" not in st.session_state:
    st.session_state.testi = []
    st.session_state.index = faiss.IndexFlatL2(1536)

def crea_embedding(testo):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=testo
    )
    return np.array(response.data[0].embedding, dtype="float32")

st.subheader("➕ Inserisci esperienza")
testo = st.text_area("Scrivi esperienza tecnica")

if st.button("Salva"):
    if testo:
        emb = crea_embedding(testo)
        st.session_state.index.add(np.array([emb]))
        st.session_state.testi.append(testo)
        st.success("Salvato!")

st.subheader("⚠️ Controlla situazione")
domanda = st.text_input("Descrivi situazione")

if st.button("Analizza"):
    if domanda and len(st.session_state.testi) > 0:
        emb = crea_embedding(domanda)
        D, I = st.session_state.index.search(np.array([emb]), k=3)

        contesto = "\n".join(
            [st.session_state.testi[i] for i in I[0] if i < len(st.session_state.testi)]
        )

        prompt = f"""
Sei un tecnico agroalimentare esperto.
Valuta il rischio e dai un consiglio pratico.

Contesto:
{contesto}

Situazione:
{domanda}
"""

        risposta = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(risposta.choices[0].message.content)
