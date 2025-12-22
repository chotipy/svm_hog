from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """
        Returns:
          boxes: List of [x, y, w, h]
          scores: List of confidence scores
        """
        raise NotImplementedError
