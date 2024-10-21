# Detection-of-Diabetic-retinopathy
This project successfully detects the diabetes by using deep learning on a fundus images and it can be used as one of method to detect the diabetes on the future. The DR stages are based on the type of lesions that appear on the retina. 


## Dataset Information

The dataset used for this project is the **MESSIDOR-2 Diabetic Retinopathy Dataset**.

### Description:
- **Number of Examinations**: 874
- **Total Images**: 1748
- **Formats**: PNG, JPG
- **Image Source**: Brest University Hospital, France, and the Messidor Program Partners

This dataset contains macula-centered eye fundus images, collected without pharmacological dilation using a Topcon TRC NW6 non-mydriatic fundus camera.

### Access the Dataset:
You can download the MESSIDOR-2 dataset from the official website:
- [MESSIDOR-2 Dataset](https://www.adcis.net/en/third-party/messidor2/)

The dataset is not hosted here due to its large size and licensing restrictions. Please refer to the provided link to obtain the data directly.

### Preprocessing Steps:
- Resize all images to a uniform size (e.g., 224x224 pixels)
- Normalize pixel values
- Augment images with techniques like rotation, zoom, etc.
