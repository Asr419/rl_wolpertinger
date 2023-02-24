from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class DocCatalogue:
    """A class to store the documents in a catalogue."""

    doc_df: pd.DataFrame
    doc_id_column: str

    def __post__init__(self) -> None:
        self.doc_df.set_index(self.doc_id_column, inplace=True)
        # sorting doc df by doc_id
        self.doc_df = self.doc_df.sort_index()

    def get_doc_features(self, doc_id: int) -> npt.NDArray[np.float64]:
        """Get the features of a document given its id."""
        return self.doc_df.loc[doc_id, :].values

    def get_docs_features(
        self,
    ) -> npt.NDArray[np.float64]:
        """Get the features of a document given its id."""
        return self.doc_df.loc[self.doc_id_column, :].values
