
# Importa las librerías necesarias
import numpy as np
import tensorflow as tf

def calculate_ssim(y_true, y_pred):
    # Asumiendo que images1 y images2 son tus dos conjuntos de imágenes para comparar
    # y ambos tienen la misma forma
    
    # Primero, determina el rango de tus datos
    min_val = np.min(y_true)
    max_val = np.max(y_true)
    
    # Prepara las imágenes para el cálculo del SSIM según su rango
    if min_val >= 0 and max_val > 1:
        # Si el rango es [0, 255], convertir a float y normalizar a [0, 1]
        y_true = y_true.astype(np.float32) / 255.0
        y_pred = y_pred.astype(np.float32) / 255.0
    # Nota: Si el rango ya es [0, 1], no se necesita ajuste adicional
    
    # Calcula el SSIM
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))
    

# Ejemplo de uso
# Asegúrate de que images1 y images2 están cargadas y tienen la misma dimensión
# Por ejemplo: images1, images2 = load_your_images()

# Calcula SSIM
# ssim_value = calculate_ssim(images

# # Ejemplo de cómo usar esta función de pérdida personalizada
# model.compile(optimizer='adam', loss=ssim_loss(max_val=255.0))