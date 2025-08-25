import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def generate_binary_image(size, index):
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)

    shapes = [
        lambda d: d.rectangle([size//4, size//4, 3*size//4, 3*size//4], outline=255, fill=255),  # Rectangle
        lambda d: d.ellipse([size//4, size//4, 3*size//4, 3*size//4], outline=255, fill=255),  # Circle
        lambda d: d.polygon([size//2, size//4, 3*size//4, 3*size//4, size//4, 3*size//4], outline=255, fill=255),  # Triangle
        lambda d: d.polygon([size//2, size//4, size//4, size//2, size//2, 3*size//4, 3*size//4, size//2], outline=255, fill=255),  # Rhombus
        lambda d: d.polygon([size//2, size//8, 5*size//8, size//3, size-1, size//3, 3*size//4, size//2, size-1, 2*size//3, 5*size//8, 2*size//3, size//2, size-1, 3*size//8, 2*size//3, 0, 2*size//3, size//4, size//2, 0, size//3, 3*size//8, size//3], outline=255, fill=255),  # Star
        lambda d: d.line([size//4, size//4, 3*size//4, 3*size//4], fill=255, width=3),  # Diagonal Lines
        lambda d: d.line([size//4, 3*size//4, 3*size//4, size//4], fill=255, width=3),  # Reverse Diagonal Lines
        lambda d: d.line([size//2, size//4, size//2, 3*size//4], fill=255, width=3),  # Vertical Lines
        lambda d: d.line([size//4, size//2, 3*size//4, size//2], fill=255, width=3),  # Horizonal Lines
        lambda d: d.arc([size//4, size//4, 3*size//4, 3*size//4], start=0, end=180, fill=255)  # Arc
    ]
    shapes[index % len(shapes)](draw)

    binary_image = np.array(image)
    return binary_image

if __name__ == '__main__':
    # Size of Image
    size = 128

    for i in range(10):
        binary_image = generate_binary_image(size, i)
        img = Image.fromarray(binary_image)
        img.save(f'active_wm_{i+1}.png')

        plt.imshow(binary_image, cmap='gray')
        plt.title(f'Binary Image {i+1} - Shape')
        plt.show()
