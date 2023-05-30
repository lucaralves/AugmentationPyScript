import imgaug.augmenters as iaa
import cv2
import os

# Path to the parent directory containing the wine image folders
parent_dir = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\Raw_Vinhos'

# List to store the wine images
wine_images = []
# List to store the augmented wine images.
augmented_images = []

if __name__ == '__main__':

    # Iterate over the wine image folders
    for wine_folder in os.listdir(parent_dir):
        wine_folder_path = os.path.join(parent_dir, wine_folder)

        # Check if the item in the parent directory is a folder
        if os.path.isdir(wine_folder_path):
            # Iterate over the images in the wine folder
            for image_file in os.listdir(wine_folder_path):
                image_path = os.path.join(wine_folder_path, image_file)

                # Read the image using OpenCV
                image = cv2.imread(image_path)

                # Append the image to the wine_images list
                wine_images.append(image)

    # Define the augmentation pipeline
    augmentation_pipeline = iaa.Sequential([
        iaa.Multiply((0.7, 1.3)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add noise
        iaa.Affine(rotate=(-10, 10)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.1))  # Crop the image
    ])
    # Define the augmentation pipeline
    augmentation_pipeline1 = iaa.Sequential([
        iaa.Multiply((0.1, 1.5)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0.2, 1.2)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0.2, 0.3 * 255)),  # Add noise
        iaa.Affine(rotate=(-5, 5)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.2))  # Crop the image
    ])
    # Define the augmentation pipeline
    augmentation_pipeline2 = iaa.Sequential([
        iaa.Multiply((0.5, 1.9)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0.5, 1.3)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0.3, 0.5 * 255)),  # Add noise
        iaa.Affine(rotate=(-5, 5)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.2))  # Crop the image
    ])
    # Define the augmentation pipeline
    augmentation_pipeline3 = iaa.Sequential([
        iaa.Multiply((0.3, 1.6)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0.7, 2.5)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0.3, 0.7 * 255)),  # Add noise
        iaa.Affine(rotate=(-3, 3)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.4))  # Crop the image
    ])
    # Define the augmentation pipeline
    augmentation_pipeline4 = iaa.Sequential([
        iaa.Multiply((0.5, 1.9)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0.6, 2.0)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0.3, 0.8 * 255)),  # Add noise
        iaa.Affine(rotate=(-4, 4)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.5))  # Crop the image
    ])
    # Define the augmentation pipeline
    augmentation_pipeline5 = iaa.Sequential([
        iaa.Multiply((0.5, 1.9)),  # Adjust brightness
        iaa.GaussianBlur(sigma=(0.8, 3.0)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0.3, 0.9 * 255)),  # Add noise
        iaa.Affine(rotate=(-7, 7)),  # Rotate the image
        # iaa.Crop(percent=(0, 0.3))  # Crop the image
    ])

    # Apply the augmentation on raw images.
    for image in wine_images:
        augmented = augmentation_pipeline.augment_image(image)
        augmented1 = augmentation_pipeline.augment_image(image)
        augmented2 = augmentation_pipeline.augment_image(image)
        augmented3 = augmentation_pipeline.augment_image(image)
        augmented4 = augmentation_pipeline.augment_image(image)
        augmented5 = augmentation_pipeline.augment_image(image)
        augmented_images.append(augmented)
        augmented_images.append(augmented1)
        augmented_images.append(augmented2)
        augmented_images.append(augmented3)
        augmented_images.append(augmented4)
        augmented_images.append(augmented5)

    # Write the augmented images on disk.
    output_dir = os.path.join(parent_dir, "Augmented_Images")
    for i, image in enumerate(augmented_images):
        output_path = f"{output_dir}\\augmented_wine_{i}.jpg"
        cv2.imwrite(output_path, image)

    print('EOF')