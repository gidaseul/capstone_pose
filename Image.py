from typing import Any

import cv2
from numpy import ndarray, dtype, generic

Image = cv2.Mat | ndarray[Any, dtype[generic]] | ndarray
