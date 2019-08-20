import unicodedata
import re


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    
    # Replace any .?! by a whitespace + the character
    # ie '!' replaced by ' !' (without quotes)
    # \1 implies first bracketed group ie [,!?]
    # r is to escape backslash
    s = re.sub(r'([,!?])',r'\1',s)
    
    # Remove any character that is not alphanumeric
    # + means one or more
    s = re.sub(r'[^a-zA-Z.!?]',r' ',s)

    # Remove a sequence of whitespace characters
    s = re.sub(r'\s+',r' ',s).strip()
    return s

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p, MAX_LENGTH=10):
    return (len(p[0].split()) < MAX_LENGTH) and (len(p[1].split()) < MAX_LENGTH)

# Filter pairs using filterPair condition (above)
def filterPairs(pairs, MAX_LENGTH = 10):
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH)]

