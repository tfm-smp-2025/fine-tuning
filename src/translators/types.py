class LLMModel:
    def invoke(self, text: str) -> str:
        raise NotImplemented('This is an abstract class, use an implemented one.')