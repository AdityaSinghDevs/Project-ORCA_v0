import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging

class ImageEmbeddingConverter:
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 model_name: str = "Facenet",
                 augmentation_count: int = 3):
        """
        Initialize the converter with configuration parameters.
        """
        self.input_size = input_size
        self.model_name = model_name
        self.augmentation_count = augmentation_count
        
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        return img

    def generate_augmentations(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate augmented versions of the input image."""
        img_array = np.expand_dims(image, axis=0)
        aug_iter = self.datagen.flow(img_array, batch_size=1)
        return [next(aug_iter)[0].astype(np.uint8) 
                for _ in range(self.augmentation_count)]

    def save_temp_image(self, image: np.ndarray, filename: str) -> str:
        """Temporarily save an image for DeepFace processing."""
        temp_path = f"temp_{filename}"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return temp_path

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from an image using DeepFace."""
        result = DeepFace.represent(
            img_path=image_path, 
            model_name=self.model_name, 
            enforce_detection=False
        )
        # Convert to numpy array and ensure it's flattened
        embedding = np.array(result[0]['embedding'], dtype=np.float32)
        return embedding.reshape(-1)  # Ensure 1D array

    def process_dataset(self, 
                       dataset_path: str, 
                       output_file: str,
                       visualize: bool = False) -> None:
        """Process entire dataset and save embeddings."""
        face_data = []
        
        for person in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person)
            
            if not os.path.isdir(person_folder):
                continue
                
            self.logger.info(f"Processing person: {person}")
            
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                
                try:
                    # Preprocess original image
                    original_img = self.preprocess_image(image_path)
                    
                    # Generate augmentations
                    aug_images = self.generate_augmentations(original_img)
                    
                    # Process original and augmented images
                    all_images = [original_img] + aug_images
                    
                    # Save temporary files and extract embeddings
                    for i, img in enumerate(all_images):
                        temp_path = self.save_temp_image(img, f"{person}_{i}.jpg")
                        
                        # Extract and format embedding
                        embedding = self.extract_embedding(temp_path)
                        # Convert embedding to list and ensure proper formatting
                        embedding_list = embedding.tolist()
                        face_data.append([person, str(embedding_list)])
                        
                        # Cleanup
                        os.remove(temp_path)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")

        # Save results with proper formatting
        df = pd.DataFrame(face_data, columns=["Person", "Embedding"])
        
        # Save with proper string formatting for embeddings
        df.to_csv(output_file, index=False)
        self.logger.info(f"Face embeddings saved successfully to {output_file}!")
        
        # Print sample for verification
        self.logger.info("\nSample of saved embeddings:")
        print(df.head(1))

if __name__ == "__main__":
    # Initialize converter
    converter = ImageEmbeddingConverter(
        input_size=(224, 224),
        model_name="Facenet",
        augmentation_count=3
    )
    
    # Process dataset
    converter.process_dataset(
        dataset_path="data/images",
        output_file="face_embeddings.csv",
        visualize=False
    )