import streamlit as st
from rag_chain import ask_book

st.set_page_config(
    page_title="দুর্গেশনন্দিনী - RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 দুর্গেশনন্দিনী (Durgeshnandini) — Knowledge Base Chatbot")
st.caption("বঙ্কিমচন্দ্র চট্টোপাধ্যায়ের উপন্যাস ভিত্তিক RAG চ্যাটবট | Powered by LangChain, ChromaDB & Llama 3.2")

with st.sidebar:
    st.header("বইয়ের তথ্য")
    st.markdown("**বই:** দুর্গেশনন্দিনী")
    st.markdown("**লেখক:** বঙ্কিমচন্দ্র চট্টোপাধ্যায়")
    st.markdown("**উৎস:** [বাংলা উইকিসংকলন](https://bn.wikisource.org/wiki/দুর্গেশনন্দিনী)")
    st.markdown("**LLM:** `deepseek-v4.1-flash-1017`")
    st.markdown("**Embedding:** `intfloat/multilingual-e5-base`")
    st.divider()
    st.subheader("নমুনা প্রশ্ন (Sample Questions)")
    sample_qs = [
        "৯৯৭ বঙ্গাব্দের নিদাঘশেষে অশ্বারোহী পুরুষ কোথা থেকে কোথায় যাচ্ছিলেন?",
        "শৈলেশ্বর মন্দিরে আশ্রয় নেওয়া যুবক নিজের কী পরিচয় দিয়েছিলেন?",
        "তিলোত্তমার পিতার নাম কী এবং তিনি কোন দুর্গের অধিপতি ছিলেন?",
        "আয়েষা কে এবং তাঁর বয়স কত ছিল?",
        "শৈলেশ্বর মন্দিরে জগৎসিংহের সাথে কাদের সাক্ষাৎ হয়েছিল?",
        "এই উপন্যাসে কি রবীন্দ্রনাথ ঠাকুরের কোনো চরিত্রের উল্লেখ আছে?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("উদ্ধৃত উৎস ও পরিচ্ছেদ দেখুন (Retrieved Context)"):
                for idx, doc in enumerate(message["sources"], 1):
                    clean_text = doc.page_content.replace("passage: ", "", 1)
                    st.markdown(f"**{idx}. {doc.metadata.get('chapter')}** ([লিংক]({doc.metadata.get('source_url')}))")
                    st.info(clean_text[:450] + "...")

user_input = st.chat_input("দুর্গেশনন্দিনী সম্পর্কে প্রশ্ন করুন (বাংলায় বা ইংরেজিতে)...")

if "prefill_question" in st.session_state:
    user_input = st.session_state.pop("prefill_question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("বইয়ের পরিচ্ছেদ অনুসন্ধান করা হচ্ছে..."):
            result = ask_book(user_input)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            with st.expander("উদ্ধৃত উৎস ও পরিচ্ছেদ দেখুন (Retrieved Context)"):
                for idx, doc in enumerate(sources, 1):
                    clean_text = doc.page_content.replace("passage: ", "", 1)
                    st.markdown(f"**{idx}. {doc.metadata.get('chapter')}** ([লিংক]({doc.metadata.get('source_url')}))")
                    st.info(clean_text[:450] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })