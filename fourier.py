import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se haya cargado correctamente
if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

# Transformada de Fourier
dft = np.fft.fft2(imagen)
#dft_shift = np.fft.fftshift(dft)  # Desplazar el componente de frecuencia cero al centro
dft_shift = dft

# Cálculo del espectro de Fourier (magnitud)
espectro = np.abs(dft_shift)
espectro_log = np.log1p(espectro)  # Aplicar transformación logarítmica

# Cálculo del ángulo de fase
fase = np.angle(dft_shift)

# Visualización de resultados
plt.figure(figsize=(12, 6))

# Imagen original
plt.subplot(1, 2, 1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

# Espectro de Fourier (magnitud)
plt.subplot(1, 2, 2)
plt.title('Espectro (logarítmico)')
plt.imshow(espectro_log, cmap='gray')
plt.axis('off')

# Ángulo de fase
#plt.subplot(1, 3, 3)
#plt.title('Ángulo de fase')
#plt.imshow(fase, cmap='gray')
#plt.axis('off')

plt.tight_layout()
plt.show()
