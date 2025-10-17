import re
TOKENS=[(r"\d+","INT"),(r"[a-zA-Z_]\w*","ID"),(r"[+\-*/]","OP"),(r"\s+",None)]
def lex(text):
    pos=0
    while pos<len(text):
        for pat,tag in TOKENS:
            m=re.match(pat,text[pos:])
            if m:
                if tag: yield(tag,m.group(0))
                pos+=m.end(); break
        else: raise SyntaxError(f"Bad char {text[pos]}")

