"""ChatGPT-style CSS theme for the Conversational RAG chat tab.

Applied via ``st.markdown(unsafe_allow_html=True)`` on top of Streamlit's native
``st.chat_message`` / ``st.chat_input`` elements. Gives user messages a
right-aligned colored bubble, assistant messages a left-aligned plain block,
compact avatars, a source card list, and a clean centered conversation column.
"""

from __future__ import annotations

CHAT_CSS = """
<style>
/* --- Centered, bounded conversation column --- */
[data-testid="stChatMessage"] {
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}

/* --- Chat bubbles --- */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding: 0.6rem 0.9rem;
    border-radius: 1rem;
    line-height: 1.55;
    font-size: 0.95rem;
}

/* User messages: right-aligned colored bubble (ChatGPT green) */
[data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
    display: flex;
    justify-content: flex-end;
}
[data-testid="stChatMessage"][data-testid="stChatMessageUser"]
  [data-testid="stChatMessageContent"] {
    background-color: var(--st-primary-color, #10a37f);
    color: #ffffff;
    border-bottom-right-radius: 0.25rem;
    max-width: 72%;
}
[data-testid="stChatMessage"][data-testid="stChatMessageUser"]
  [data-testid="stChatMessageContent"] a {
    color: #e6f7f2;
}

/* Assistant messages: left-aligned plain block */
[data-testid="stChatMessage"][data-testid="stChatMessageAssistant"] {
    display: flex;
    justify-content: flex-start;
}
[data-testid="stChatMessage"][data-testid="stChatMessageAssistant"]
  [data-testid="stChatMessageContent"] {
    background-color: var(--st-secondary-background-color, #f7f7f8);
    border-bottom-left-radius: 0.25rem;
    max-width: 88%;
}

/* --- Avatars --- */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
    width: 2rem;
    height: 2rem;
    font-size: 1.1rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--st-secondary-background-color, #f7f7f8);
}

/* --- Typography inside messages --- */
[data-testid="stChatMessageContent"] p {
    margin-bottom: 0.35rem;
}
[data-testid="stChatMessageContent"] pre {
    border-radius: 0.5rem;
    font-size: 0.85rem;
}
[data-testid="stChatMessageContent"] code {
    font-size: 0.85rem;
}

/* --- Source card list --- */
div[data-testid="stChatMessage"] details[data-testid="stExpander"] {
    border: 1px solid var(--st-secondary-background-color, #f7f7f8);
    border-radius: 0.75rem;
    background-color: rgba(247, 247, 248, 0.5);
    margin-top: 0.4rem;
    max-width: 88%;
}
div[data-testid="stChatMessage"] details[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--st-primary-color, #10a37f);
}
div[data-testid="stChatMessage"] details[data-testid="stExpander"] ul {
    margin-bottom: 0.25rem;
}
div[data-testid="stChatMessage"] details[data-testid="stExpander"] li {
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}

/* --- Chat input bar (sticky at bottom of the tab) --- */
div[data-testid="stChatInput"] {
    border: 1px solid #e5e7eb;
    border-radius: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--st-primary-color, #10a37f);
    box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.15);
}

/* --- Copy / action buttons under assistant messages --- */
button[data-testid="stChatCopyButton"] {
    font-size: 0.75rem;
}

/* --- Follow-up suggestion chips --- */
button[data-testid="stButton"] {
    border-radius: 999px;
    border: 1px solid #e5e7eb;
    background-color: var(--st-secondary-background-color, #f7f7f8);
    font-size: 0.8rem;
    padding: 0.3rem 0.8rem;
    min-height: 2rem;
}
button[data-testid="stButton"]:hover {
    border-color: var(--st-primary-color, #10a37f);
    color: var(--st-primary-color, #10a37f);
}
</style>
"""


def apply_chat_theme() -> None:
    """Inject the ChatGPT-style CSS into the current Streamlit app."""
    import streamlit as st

    st.markdown(CHAT_CSS, unsafe_allow_html=True)
