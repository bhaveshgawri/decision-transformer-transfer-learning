import imageio
from IPython.display import Image, display
from io import BytesIO

from pyvirtualdisplay import Display

class GIFMaker:
    def __init__(self):
        self.reset()
        display = Display(visible=0, size=(1280, 720))
        display.start()

    def reset(self):
        self.images = []
        self.buffer = BytesIO()
  
    def append(self, img):
        self.images.append(img)

    def display(self):
        imageio.mimsave(self.buffer, self.images, format='gif')
        gif = Image(data=self.buffer.getvalue())
        display(gif)
        return gif
    
    def save(self, file_name):
        imageio.mimsave(f'./cache/outputs/{file_name}.gif', self.images, format='gif')

    def __len__(self):
      return len(self.images)
    