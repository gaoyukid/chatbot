import corenlp
from .happyfuntokenizer import Tokenizer

class HappyFunTokenizer(Tokenizer, corenlp.Annotator):
    def __init__(self, preserve_case=False):
        Tokenizer.__init__(self, preserve_case)
        corenlp.Annotator.__init__(self)

    @property
    def name(self):
        """
        Name of the annotator (used by CoreNLP)
        """
        return "happyfun"

    @property
    def requires(self):
        """
        Requires has to specify all the annotations required before we
        are called.
        """
        return []

    @property
    def provides(self):
        """
        The set of annotations guaranteed to be provided when we are done.
        NOTE: that these annotations are either fully qualified Java
        class names or refer to nested classes of
        edu.stanford.nlp.ling.CoreAnnotations (as is the case below).
        """
        return ["TextAnnotation",
                "TokensAnnotation",
                "TokenBeginAnnotation",
                "TokenEndAnnotation",
                "CharacterOffsetBeginAnnotation",
                "CharacterOffsetEndAnnotation",
               ]

    def annotate(self, ann):
        """
        @ann: is a protobuf annotation object.
        Actually populate @ann with tokens.
        """
        buf, beg_idx, end_idx = ann.text.lower(), 0, 0
        for i, word in enumerate(self.tokenize(ann.text)):
            token = ann.sentencelessToken.add()
            # These are the bare minimum required for the TokenAnnotation
            token.word = word
            token.tokenBeginIndex = i
            token.tokenEndIndex = i+1

            # Seek into the txt until you can find this word.
            try:
                # Try to update beginning index
                beg_idx = buf.index(word, beg_idx)
            except ValueError:
                # Give up -- this will be something random
                end_idx = beg_idx + len(word)

            token.beginChar = beg_idx
            token.endChar = end_idx

            beg_idx, end_idx = end_idx, end_idx

annotator = HappyFunTokenizer()
# Calling .start() will launch the annotator as a service running on
# port 8432 by default.
annotator.start()

# annotator.properties contains all the right properties for
# Stanford CoreNLP to use this annotator.
with corenlp.CoreNLPClient(properties=annotator.properties, annotators="happyfun ssplit pos".split()) as client:
    ann = client.annotate("RT @ #happyfuncoding: this is a typical Twitter tweet :-)")

    tokens = [t.word for t in ann.sentence[0].token]
    print(tokens)