# Udacity Project 2: Image Classifier by Niloy Deb Barma

This project is part of the Udacity Nanodegree program and focuses on developing an image classifier using deep learning techniques. The goal is to create a model capable of classifying images into different categories, specifically using the [102 Flower Categories dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) to train and evaluate the model.

# Verified Certificate Of Nanodegree Program Completion
## AI Programming with Python

[AI Programming with Python](https://www.udacity.com/certificate/e/aaa90186-2dce-11ef-889a-97c882f2afe3)

### Certificate Overview
This certificate confirms the completion of the AI Programming with Python Nanodegree program. The program covers:

- **Fundamentals of Python programming**
- **Working with data using NumPy and Pandas**
- **Introduction to machine learning with scikit-learn**
- **Building a neural network using PyTorch**

### Skills Acquired
- Python programming
- Data manipulation and analysis
- Machine learning fundamentals
- Neural network development

For more details, please visit the [official program page](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

## Repository Structure

```
Udacity-Project-2-my-own-image-classifier/
├── GitHub/workflows
├── assets
├── flowers
│   ├── train
│   ├── valid
│   └── test
├── README.md
├── Udacity-2nd-Image-Classifier(Use Google Colab).ipynb
├── cat_to_name.json
├── image_classifier_project.ipynb
├── predict.py
└── train.py
```

## Getting Started

### Prerequisites

- [Google Colab](https://colab.research.google.com/) account
- Basic knowledge of Python and PyTorch
- A GPU runtime in Google Colab (preferably T4 GPU)

### Using Google Colab

1. **Open Google Colab:**
   - Navigate to [Google Colab](https://colab.research.google.com/).
   - Sign in with your Google account.

2. **Upload the Notebook:**
   - Click on the `File` menu and select `Upload notebook`.
   - Upload the provided `.ipynb` file: `Udacity-2nd-Image-Classifier(Use Google Colab).ipynb`.

3. **Set Up GPU Runtime:**
   - Go to the `Runtime` menu.
   - Select `Change runtime type`.
   - Under `Hardware accelerator`, choose `GPU`.
   - Click `Save`.

4. **Ensure T4 GPU is Selected:**
   - Google Colab typically provides a T4 GPU by default. However, if another GPU is assigned, you can try reloading the runtime until a T4 GPU is allocated.

### Running the Notebook

1. **Install Dependencies:**
   - The notebook includes cells to install necessary dependencies such as `torch`, `torchvision`, and `PIL`. Ensure these cells are executed before running the rest of the notebook.

2. **Load and Preprocess Data:**
   - Follow the steps in the notebook to load and preprocess the dataset.

3. **Train the Model:**
   - Use the cells provided to train the model. Monitor the training process and adjust hyperparameters as needed.

4. **Evaluate the Model:**
   - Evaluate the trained model using the validation and test datasets. The notebook includes cells to perform these evaluations and display results.

5. **Make Predictions:**
   - Use the trained model to make predictions on new images. The notebook provides functions to process images and make predictions.

6. **Save and Load Checkpoints:**
   - Save the trained model checkpoints and load them later for inference. The notebook includes cells to handle checkpoint saving and loading.

### Scripts

- `train.py`: Script to train the image classifier model.
- `predict.py`: Script to make predictions using the trained model.
- `image_classifier_project.ipynb`: Jupyter Notebook with the complete project workflow.
- `Udacity-2nd-Image-Classifier(Use Google Colab).ipynb`: Updated notebook specifically designed for Google Colab.

### Dataset

- `flowers/`: Directory containing training, validation, and test datasets.
- `cat_to_name.json`: JSON file mapping category labels to flower names.

## Acknowledgements

- This project is part of the Udacity Nanodegree program.
- The dataset used is the [102 Flower Categories dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) provided by the University of Oxford.

## About Niloy Deb Barma

Niloy Deb Barma is a dedicated and passionate AI enthusiast with a focus on leveraging technology to create a positive impact. With a strong background in AI and ML, Niloy has been involved in various projects and research, continuously pushing the boundaries of innovation.

## Contact

For any questions or issues, please feel free to reach out to Niloy Deb Barma through [LinkedIn](https://www.linkedin.com/in/niloydebbarmacpscr).

---

Happy coding!
