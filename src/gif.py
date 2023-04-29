import imageio
from IPython.display import Image, display
from io import BytesIO

from pyvirtualdisplay import Display

import numpy as np

class GIFMaker:
    def __init__(self) -> None:
        self.reset()
        display = Display(visible=0, size=(1280, 720))
        display.start()

    def reset(self) -> None:
        self.images = []
        self.buffer = BytesIO()
  
    def append(self, img: np.ndarray[int]) -> None:
        self.images.append(img)

    def display(self) -> Image:
        imageio.mimsave(self.buffer, self.images, format='gif')
        gif = Image(data=self.buffer.getvalue())
        display(gif)
        return gif
    
    def save(self, file_path: str) -> None:
        imageio.mimsave(file_path, self.images, format='gif')

    def __len__(self) -> int:
      return len(self.images)
    