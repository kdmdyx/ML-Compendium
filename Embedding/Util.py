def tokenize(rs: str):
    # Fileter punctuations by character
    # Using set for lookup and temp string for theta(n) runtime
    filtered = ''
    punc = {',', '.', ';', ':', "'", "\"", '!'}
    for cha in rs:
        if cha not in punc:
            filtered.join(cha)
    
    # Split into list of vocabs
    [token.lower() for token in filtered.split(' ')]


def buildVocab(rs: str):
    # Extract tokens
    tokens = tokenize(rs)

    # Extract unique vocabs and sort
    vocabs = [set(tokens)].sort()

    # map each vocab to its index number
    diction = { vocab:i for vocab, i in enumerate(vocabs, 0)}

    return diction


