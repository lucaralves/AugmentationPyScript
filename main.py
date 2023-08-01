import imgaug.augmenters as iaa
import cv2
import os
import glob

# Path to the parent directory containing the retail image folders
retail_folder_path = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\Google_Cloud\Temporario'

# List to store the wine images
retail_images = []
# List to store the augmented retail images.
augmented_images = []

if __name__ == '__main__':


    # Iterate over the images in the retail folder
    for image_file in os.listdir(retail_folder_path):
        image_path = os.path.join(retail_folder_path, image_file)

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Append the image to the retail_images list
        retail_images.append(image)

    # Define the augmentation pipelines
    augmentation_pipelines = [
        iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),  # Adjust brightness
            iaa.GaussianBlur(sigma=(0.0, 0.5)),  # Apply Gaussian blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)),  # Add noise
            iaa.Affine(
                rotate=(-5, 5),  # Rotate the image
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Translate the image
                scale=(0.9, 1.1),  # Scale the image
                shear=(-5, 5)  # Shear the image
            ),
            iaa.Crop(percent=(0, 0.05))  # Crop the image
        ]),
        iaa.Sequential([
            iaa.Multiply((0.9, 1.1)),  # Adjust brightness
            iaa.GaussianBlur(sigma=(0.0, 0.3)),  # Apply Gaussian blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # Add noise
            iaa.Affine(
                rotate=(-3, 3),  # Rotate the image
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},  # Translate the image
                scale=(0.95, 1.05),  # Scale the image
                shear=(-3, 3)  # Shear the image
            ),
            iaa.Crop(percent=(0, 0.03))  # Crop the image
        ]),
        iaa.Sequential([
            iaa.Multiply((0.7, 1.0)),  # Adjust brightness
            iaa.GaussianBlur(sigma=(0.0, 0.2)),  # Apply Gaussian blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),  # Add noise
            iaa.Affine(
                rotate=(-4, 4),  # Rotate the image
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},  # Translate the image
                scale=(0.95, 1.05),  # Scale the image
                shear=(-2, 2)  # Shear the image
            ),
            iaa.Crop(percent=(0, 0.02))  # Crop the image
        ]),
        iaa.Sequential([
            iaa.Multiply((0.6, 1.2)),  # Adjust brightness
            iaa.GaussianBlur(sigma=(0.0, 0.4)),  # Apply Gaussian blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)),  # Add noise
            iaa.Affine(
                rotate=(-3, 3),  # Rotate the image
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},  # Translate the image
                scale=(0.95, 1.05),  # Scale the image
                shear=(-4, 4)  # Shear the image
            ),
            iaa.Crop(percent=(0, 0.04))  # Crop the image
        ])
    ]

    # Apply the augmentation on retail images.
    for image in retail_images:
        augmented_images.append(image)
        for pipeline in augmentation_pipelines:
            augmented = pipeline.augment_image(image)
            augmented_images.append(augmented)

    # Write the augmented images on disk.
    output_dir = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\Google_Cloud\continente-produtos-aumentados'

    # Lista todos os arquivos no diretório especificado
    arquivos = glob.glob(os.path.join(output_dir, '*'))

    # Filtra apenas os arquivos com as extensões de imagens que deseja contar
    extensoes_imagens = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Adicione mais extensões se necessário
    imagens = [arquivo for arquivo in arquivos if os.path.splitext(arquivo)[1].lower() in extensoes_imagens]

    for i, image in enumerate(augmented_images):
        output_path = f"{output_dir}\\augmented_product_{len(imagens) + i}.jpg"
        cv2.imwrite(output_path, image)