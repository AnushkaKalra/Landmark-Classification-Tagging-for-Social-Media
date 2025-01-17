# Landmark Classification & Tagging for Social Media

## 🌍 **Project Overview**
This project aims to classify and tag landmarks in user-uploaded images for social media platforms. 

---

## 🚀 **Features**
- **Custom CNN Model:** Designed from scratch to establish a baseline for landmark classification.
- **Transfer Learning:** Fine-tuned the *ResNet18 model* to significantly boost classification accuracy and reduce training time.
- **Data Augmentation:** Enhanced the dataset using techniques like flipping, rotation, and scaling to improve model generalization.
- **Interactive Application:** Developed a simple web app to upload images and visualize predictions.
- **Real-Time Tagging:** Displays the top 5 predictions with confidence scores for each image.

---

## 🛠️ **Tech Stack**
### **Programming Languages:**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### **Libraries & Frameworks:**
 ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
 

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

![Seaborn](https://assets.streamlinehq.com/image/private/w_150,h_150,ar_1/f_auto/v1/icons/2/seaborn-mazs5fsvs6lluqsnmeik89.png/seaborn-b4pddoh3hfg4k85ug0oazo.png?_a=DAJFJtWIZAAC)

### **Tools:**
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


---

## 📂 **Dataset and Preprocessing**
- A labeled dataset of landmark images was used to train and validate the models.
- Preprocessing steps included:
  - Resizing images to consistent dimensions.
  - Normalizing pixel values to optimize model convergence.
  - Data augmentation using `torchvision.transforms` to expand dataset diversity.

---

## 🧠 **Model Development**
1. **Custom CNN:**
   - Designed a CNN with convolutional, pooling, and fully connected layers to classify images.
   - Served as the baseline model for initial experimentation.

2. **Transfer Learning with ResNet18:**
   - Utilized the pre-trained ResNet18 model from Torchvision.
   - Fine-tuned the final layers to adapt to the landmark classification task.
   - Achieved a significant boost in test accuracy compared to the custom CNN.

---

## 📊 **Performance Metrics**
- **Custom CNN Accuracy:** Baseline accuracy established with iterative improvements.
- **Transfer Learning Accuracy:** Achieved over **60% test accuracy** with ResNet18.
- **Evaluation Metrics:**
  - Cross-entropy Loss

---

## 🎯 **Key Outcomes**
- Improved classification performance using transfer learning.
- Demonstrated effective preprocessing and model optimization strategies.

---

## 📚 **Learning Highlights**
- Learnt **transfer learning** techniques for image classification tasks.
- Enhanced proficiency in PyTorch for building, training, and fine-tuning models.
- Gained hands-on experience in data preprocessing and augmentation using `torchvision.transforms`.




---

## 🤝 **Contributions**
This project was developed individually as part of Udacity's AWS Machine Learning Fundamentals Program.
Special thanks to Udacity for the resources and mentorship provided during the course.

---

## 📬 **Contact**
Feel free to reach out with any questions or feedback:
- **LinkedIn:** [Anushka Kalra](https://www.linkedin.com/in/https://www.linkedin.com/in/anushka-kalra-300286213/)
- **GitHub:** [Anushka Kalra](https://github.com/AnushkaKalra)
