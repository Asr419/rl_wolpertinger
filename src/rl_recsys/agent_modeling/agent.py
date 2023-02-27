import abc


class AbstractSlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    pass

    def __init__(
        self,
        slate_gen_func,
    ) -> None:
        self.slate_gen_func = slate_gen_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    @abc.abstractmethod
    def init_state(**kwargs) -> torch.Tensor:
        """Initialize the first estimated state of the agent"""
        pass
