def coloredSentences(sentences, out=15):
    """
    Display in green the positive sentences, and in red the negative sentences
    - sentences is a dict
        - sentences.keys() are the sentences to display
        - sentences.values() are booleans that encode the sentiment
    - out is an integer indicating the maximum number of sentences to display
    """
    cpt = 0

    for _, w in enumerate(sentences):
        cpt += 1
        if sentences[w]:
            print(Fore.GREEN + w)
        else:
            print(Fore.RED + w)

        if cpt > out:
            break