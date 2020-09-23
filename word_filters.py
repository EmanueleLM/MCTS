"""
Define filters to exclude some kind of neighbors from the results.
All the filter are compliant with Neighbors class (Neighbors.py) and its filter's functions.
"""

def nltk_filter(w, candidates, granularity='exact', auxiliary_list=None):
    """
    Nltk filter: given as input a list of words, it tags them with the function
     nltk.pos_tag and discard those which are out of context with different granularities.
    w: string
        Word whose type is used for filtering.
    candidates: list
        List of words to filter, e.g. ['good', 'bad', 'better'].
    granularity: string
        optional: 'exact' will take in the result list only the words whose type is exactly 
            the same as input w, while 'smooth' will take all the words that are classified with
            an acronym that shares the first 2 letters (e.g. as 'go' is classified as VRB and 'went'
            as VRBP (past), with granularity='exact' we discard 'went', while we take it with 
            granularity='smooth'). It is possible to define a cutom file where to specify which category
            can be included and which one excluded. Modify 'filters/nltk.cfg'.
    neighbors_filters: list
        optional: list of functions to filter the results: each function must recieve as input a list of words, 
            and outputs a filtered list (for example discarding words that are tagged differently by nltk.pos_tag function)
    auxiliary_list: list
        optional: list that has the same size of candidates and is used to return couples of (word, element) where also
            elements in auxiliary_list are filtered.

    Returns
    -------
        list of words filtered by their type plus the auxiliary list if auxiliary_list is not None.
    """
    from nltk import pos_tag, tokenize
    categories_allowed = {}
    if granularity == 'custom':
        lines = [line.rstrip('\n') for line in open("./filters/nltk.cfg","r")]
        for l in lines:
            c, tmp = l.replace(' ', '').split(':')
            categories_allowed[c] = tmp.split(',')
    category = pos_tag([w])[0][1]  # extract category as the output is in the form [(<word>, <category>)]
    tagged_candidates = pos_tag(candidates)
    filtered_words, auxiliary_elements = [], []
    for t,i in zip(tagged_candidates, range(len(tagged_candidates))):
        if granularity == 'exact':
            if t[1] == category:
                filtered_words.append(t[0])
        elif granularity == 'smooth':
            if t[1][:2] == category[:2]:
                filtered_words.append(t[0]) 
        elif granularity == 'custom':
            if t[1] in categories_allowed[category]:
                filtered_words.append(t[0])
        else:
            raise Exception("Not Implemented Exception: we are working on it.")
        if auxiliary_list != None:
            auxiliary_elements.append(auxiliary_list[i])
    return (filtered_words if auxiliary_list is None else (filtered_words, auxiliary_elements))
