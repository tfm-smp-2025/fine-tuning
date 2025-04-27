class LLMModel:
    def invoke(self, messages: list[str]) -> str:
        raise NotImplementedError("This is an abstract class, use an implemented one.")
