import unicodedata

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # Replace any .?! by a whitespace + the character
    # ie '!' replaced by ' !' (without quotes)
    # 