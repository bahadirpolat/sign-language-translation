LLAMA3_CONTEXT_TRANSLATION_PROMPT = """<s>[INST] <<SYS>>
You are sign language translator. Your task is to translate the {} into {} text.
If context is given, you should consider the context while translating the sign language.
Context: {}
<<SYS>>
{} Sign Language: {}
"""

T5_CONTEXT_TRANSLATION_PROMPT = """Translate {} into {} text. {} Sign Language: {}""" 