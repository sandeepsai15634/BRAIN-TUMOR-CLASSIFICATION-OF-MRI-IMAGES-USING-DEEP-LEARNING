BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING

Abstract

Brain Tumor is the abnormal growth of mass of the cells in and around the brain different types of brain tumors exist the ability to diagnose a brain tumor by a radiologist is dependent on his or her level of experience. early detection or classification of brain tumors and diagnosis is a critical process early detection of tumors can prevent from surgery the most common technique used to diagnose Brain Tumor is Magnetic resonance imaging (MRI) which provides a detailed information of the internal structure in the human body. in recent years most advancement in medical imaging and deep learning architectures lead for the automation in the classification of brain tumors. A crucial aspect of medical image processing is the categorization of images of brain tumors. It assists healthcare providers identify and manage individuals correctly. One of the primary methods for visualizing brain tissue is magnetic resonance imaging (MR). 

classifying brain tumor MR images. In these study various techniques are proposed for classification of brain tumors as meningioma, glioma and pituitary tumor using the deep learning frameworks such as convolutional neural network (CNN) ResNet50 and VGG16. and we also combined different models using transfer learning. where it's far a famous method in deep mastering. the model already advanced for a venture is utilized in another venture. in our study We combined CNN with ResNet50 and VGG16 The Framework which includes a data set of 7033 simulated images from Kaggle are used for the study. It is essential to comprehend brain diseases, such as brain tumors (BT), in order to evaluate the tumors and help patients receive the appropriate treatment based on their classifications. There are many imaging schemes available for BT detection, like Magnetic Resonance Imaging (MRI), which is typically used due to its higher image quality. The experimental findings demonstrate how effective the suggested method is for BT multi-class classification.

A brain tumor is an unusual mass of cells that grows inside and around the brain. Brain tumors are typically classified into two types according to the location and growth of the brain's cells. The growth and location of the cells in a brain tumor typically lead to two types of brain tumors; the signs and symptoms of a mind tumor can also differ based on the location of the tumor. The advancement of technology has made it possible to diagnose brain masses or tumors using magnetic resonance imaging (MRI) and machine learning. This study proposes a newly constructed transfer deep-learning model to help identify brain cancers early into subclasses such as pituitary, meningioma, and glioma. Many layers of isolated convolutional neural network (CNN) models are first built from scratch in order to evaluate their performance on brain magnetic resonance imaging (MRI).
Next, employing the transfer-learning principle, the 22-layer, binary-classification (tumor or no tumor) isolated-CNN model is used again to modify the weights of the neurons in order to classify brain MRI images into tumor subclasses.

Keywords--Brain Tumor; CNN Architecture; Deep Learning; Classification; ResNet50; VGG16; Transfer learning. 

Introduction

A brain tumor is a collection of abnormal cells in our brain detection and classification of brain tumors is an important aspect to suggest the appropriate treatment in avoidance of critical surgery in an early stage. Less than 2% of cases of cancer in humans are brain cancer, according to the World Health Organization's (WHO) cancer report. brain tumors can be cancerous(malignant) or non-cancerous(benign) when these tumors grow, they can contribute to the increase in the mass by creating the pressure inside the skull this can be harmful and also it can be life –threatening. A brain tumor is an illness that starts in the brain's tissues. Brain tumors are caused by the abnormal proliferation and invasion of malignant cells on brain tissues. Whereas secondary brain cancers develop outside of the brain and ultimately spread there, primary brain tumors originate inside the brain cells. MRI scans, a very advanced medical imaging method, have shown to be the finest tool to date for analyzing brain tumors, which can be classified as malignant or non-malignant. MRI scans are better to other imaging methods because they may create 3D data with a strong contrast between the soft tissues. Many deep learning approaches have been used to analyze these MRI data.
Benign tumors are mostly non-cancerous as they grow slowly and do not typically spread to other parts of the body whereas malignant tumors are cancerous, they grow rapidly and can easily spread to other parts of the brain of central nervous system.

![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/809fe330-2596-41a9-be2f-b17b0bee8c7e)

                  MR Images of Malignant Tumor

![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/29881764-ec3a-43f6-bd53-f1139bccf480)

                  
                  MR Image of Benign Tumor

                  
Brain tumors are broadly classified into primary brain tumors and secondary brain tumors. primary brain tumors are originated in the brain and secondary brain tumors are developed in any part of the body and spread to brain. This study focused on mainly Three Types of primary tumors such as glioma, meningioma and pituitary tumors. glioma are tumors which are developed from the glial cells meningioma tumors that are arise from the meninges- the membranes that surrounds the spinal cord and brain. Pituitary tumors are unusual growths that develops in the pituitary gland.

                  

Methodology:

Methodology begins by taking brain MRI images of tumor patients. for this study images have been chosen from Kaggle the dataset is all about the MRI scans of patients with tumors as well as without tumors. All brain MRI images contain unwanted data in the form of noise, which lead to the lower performance of the model therefore it is required to remove the unwanted noise. The highest point is to determine using the cropping method the noise in the data is removed using two techniques such as breakdown and inflation process the weight, height and size of all the MRI scans of the images is not the same all the images were resized to 224 X 224 in order to normalize all the images into one size, then all the images were normalized and scaled before fed into the model. Because they aren't designed with image processing in mind, conventional neural networks require small amounts of input.


The CNN processes sensory images in humans and other animals by arranging its neurons in a pattern resembling the frontal lobe. CNN makes use of multilayer perceptron technology, which is designed with low processing requirements in mind. Convolutional, pooling, and related layers are the three most prevalent layers in CNN. Based on methodological assessment, there isn't a related study that looks at the performance of CNN models with TL on different versions of anomalous MRI datasets.The neural network image input layer stores the image that will be processed as input. The first layer of artificial neurons processes the data that is brought in by these artificial input neurons. An image's width, height, and depth correspond to those of its input layer. The number of channels in an image is indicated by its depth; for instance, a grayscale image has one channel, but an RGB image has three. The input layer is followed by the convolution layer, which is the central layer of CNN. This layer applies the dot product to the whole input volume depth and has a tiny receptive field called a filter or kernel. 


An output data volume integer called a feature map is produced by this process. The next accessible arena of the same input picture is then traversed by the kernel in a stride, and the dot products are calculated once more between the new accessible arena and the same kernel. Ultimately, a feature map is created from the data.
The amount of data utilized for training greatly affects how well a CNN can classify images. After a few iterations, the CNN model will begin fitting again if the dataset is small. Under these conditions, the DL subdomain known as TL has developed and shown promise in the field of medical image diagnosis [13]. It is possible that the final dense layers were trained using data from updated classes in TL. Although CNNs have improved to address the aforementioned issues, overfitting for small datasets remains a problem. In order to prevent overfitting, DCNN models undergo pre-training. 


A CNN typically can be divided into two major parts known as feature extraction and classification. The architecture of CNN as mainly five layers such as (input, convolutional, polling, fully connected and output layer) for future extraction from the images convolutional and pooling layers are used whereas for classification or prediction. The fully connected layers and output layers are utilised. in the study CNN was developed to classify brain MRI images into different classes such as glioma, meningioma and pituitary. the isolated CNN with different layers were built for this task The detailed about the structure and parameters of the CNN architecture for multi class classification are given in the table.


![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/631fea6d-4073-4880-b6c7-063db86adf25)


            framework to illustrate the classification of tumor in the brain MRI images.



Dataset used:

There are 7033 MRI images in the dataset, of which 5722 are training dataset and rest 1311 are belonging to testing dataset of which belonging to various classes like glioma, meningioma and pituitary tumor. The image dataset used in this research is an open-source dataset available on Kaggle.


Convolutional Neural Networks:

Convolutional neural network is a type of multilayer neural network that is meant to recognize visual patterns from images in CNN the term convolutional referred to as mathematical operation in which you can obtain a new function by multiplying two functions. in the simple words to matrix represents the two images the product of the two Matrix provides the output that is used to extract Useful information from the input image. CNN is different from other convolutional neural networks by a sequence of convolutional layers the add a complexity to the existing equation and CNN cannot function properly without conventional layers. There are multiple layers in the Convolutional Neural Network (CNN) model utilized in this work, including a convolution layer, a flatten layer, a dropout layer, a pooling layer, and a dense layer. Convolutional layers can identify repeating patterns in an image by applying a filter that is slid over the image input and creating a feature map. The Classification algorithm's classifier performs less well due to the utilization of pooling layers, which introduce invariance by down sampling the retrieved features.


After the operation's results are known, the model will execute the activation and data pooling processes. The feature map's size may be reduced as a result of the Pooling layer process. A feature map produced by convolution is sent back into subsequent rounds of the procedure. Following that, a completely linked layer process is finished using a vector-based flatten feature map, which eventually results in an image classification. With adaptive learning rates, the training algorithm may monitor the model's performance and change the learning rate as needed in real time. Three minutes should be the maximum for the 100 Epoch train time. Small amounts of data are required for conventional neural networks because they are not particularly designed for processing images.


Neurons in the CNN are arranged in a way resembling the frontal lobe, processing sensory images in humans and other animals. Neuron layers covering the entire visual field are arranged to remove the disadvantage of fragmented processing. CNN makes use of multilayer perceptron-like technologies designed to handle light computational loads. Convolutional, pooling, and related layers are the three most prevalent layers in a  convolutional neural network. The Essential Components of CNN for Brain Tumor Identification. The amount of data utilized for training greatly affects how well a CNN can classify images. After a few iterations, the CNN model will begin fitting again if the dataset is small. Under these conditions, the DL subdomain known as TL has developed and shown promise in the field of medical image diagnosis. It is possible that the final dense layers were trained using data from updated classes in TL. Although DCNNs have improved to address the aforementioned issues, overfitting for small datasets remains a problem. Few research were done on brain MRI classification tasks, and CNN models are pre-trained using TL to solve the overfitting problem. A methodological assessment indicates that there is a lack of a comparable study evaluating the performance of CNN models with TL on different versions of anomalous MRI datasets.


The neural network image input layer stores the input image for processing. It is composed of artificial neuron layers that process the first layer of data, which is brought in by artificial input neurons. An image's width, height, and depth should match the dimensions of its input layer. An image's depth reveals how many channels it has; a grayscale image has one channel, but an RGB image has three. The input layer is followed by the convolution layer, which is the central layer of CNN. A methodological assessment indicates that there is a lack of a comparable study evaluating the performance of CNN models with TL on different versions of anomalous MRI datasets.


![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/a37a9d02-573c-4dd6-80a5-8175fbafaf35)


                    Layers in CNN Architecture


![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/cf35bce3-3fc3-47e8-864c-5db89a91cdb8)

                   convolution operations on input image

                   
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/35210488-6416-4fa4-9630-1434878d0f39)


                  Pooling operations


RESNET50:


A CNN resnet stands for residual network variant known as ResNe50 points to the residual blocks that transform the network's architecture into one based on hundreds of layers, enabling the training of extremely deep networks. The gradient Descent problem is a typical deep learning framework concern that arises as the number of layers in the network increases. This issue causes gradients to grow unnecessarily small or huge, which has a detrimental effect on training as the number of layers increases. Training and testing are both impacted by the increase in error rate.

A particular kind of neural network architecture known as ResNet was created especially for very deep networks—that is, networks with hundreds or even thousands of layers—which were previously difficult to train because of the vanishing gradient issue. 

A phenomenon known as disappearing gradient arises when deep neural networks, especially ones with numerous layers, are being trained. The issue occurs when, during training, the parameters in the network's earlier layers are backpropagated from the output layer to the input layer, resulting in a very small differential value of the loss function with regard to those parameters. 

This may result in the early layers learning very slowly or not at all, which would affect the network's performance. Various solutions have been devised to address the vanishing gradient issue, such as the utilization of substitute activation functions, normalizing strategies, and enhanced weight initialization procedures. The ResNet design presents a revolutionary technique known as "residual learning," which allows the network to bypass some levels and discover the underlying mapping of inputs to outputs. 

In order to achieve this, shortcut connections, also known as "skip connections," are introduced. These connections enable the network to propagate the input directly to a deeper layer within the network, thereby avoiding one or more levels.

A startling finding in deep learning research led to the development of the network structure. They have trained networks with up to 152 layers using the ResNet architecture. Create this sentence in a unique manner. 
They have trained networks with up to 152 layers using the ResNet architecture. Create this sentence in a unique manner. 

They have trained networks with up to 152 layers using the ResNet architecture. create this sentence in a unique manner. The Resnet50 multi-layer deep CNN framework for the transfer learning process is presented in this experimental work.
 This model is a pre-defined technique for identifying brain tumors on various images from the ImageNet collection. This gives ImageNet the initial weights for the suggested deep neural network, which are pre-trained weights.

 ![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/802dbd21-e924-43fb-8666-f26f2d9977ec)

                    Layers in Resnet Architecture



![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/0474b104-85f4-4240-ac7f-c5210c7381b6)

                    Skip connection in ResNet

RESULTS AND DISCUSSIONS:


We experimented with an MRI scan for our investigation. The input photos are first scaled to 224 by 224. The preprocessing step then turns it into a grayscale image. Following that, we ran a different set of hyper-parameters and used the CNN model to get the best result between precision and efficiency. Although this approach is straightforward, it necessitates managing the model's hyper-parameters, particularly the initial learning rate. It is crucial to optimization since it establishes the rate at which weights are changed to get the lowest loss function.

CNN architectures employed the SoftMax function as a classification layer and the transfer learning approach to classify the dataset. Furthermore, 224 × 224 pixels was the input size for the other CNN architectures. Additionally, the SoftMax function was used to classify the features that were obtained from the final layers of each CNN that provided features. Table lists the values of the parameters that the CNNs employed.


Model configurations and hyperparameters:

![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/186230f8-e146-4091-975d-840f5d5fac36)


LOSS FUNCTION ACCURACY :

CNN:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/f7f0551a-9c48-4db5-b569-0ce1af446f00)

VGG 16:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/069313ba-2691-4301-b331-b34343ca4f76)

VGG 16 TRANSFER LEARNING:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/4bcec924-8b38-4f47-898a-99b244966446)

ResNet 50:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/e69bb4de-eb18-4760-b7e2-e21032a665cc)

ResNet 50 TRANSFER LEARNING:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/f7d7166a-6386-4955-b628-c41f596126e1)

CLASSIFICATION REPORT:
![image](https://github.com/sandeepsai15634/BRAIN-TUMOR-CLASSIFICATION-OF-MRI-IMAGES-USING-DEEP-LEARNING/assets/119305751/20360422-d9c2-4c8d-9a46-69c261b3d268)

CONCLUSION:

The application of deep learning algorithms to the categorization of brain tumors is covered in this study. This work assessed many deep learning techniques, including CNN, ResNet50, and VGG16, and compared the proportionate outcomes to identify the best CNN model for classifying cancers from MRI scans. Among the models in this work, the top performing model, CNN, displayed the highest accuracy, at 98.82%. Using MRI image datasets, the suggested methodologies can also be expanded to classify other tumor forms, such as gliomas, meningiomas, and pituitary tumors. Ultimately, this study finds that CNN outperforms the other two assessed classifiers in terms of accuracy, loss, and validation loss.

The recommended model achieved the best classification accuracy with the least amount of computing work after training the framework. This means that brain malignancies can be properly classified in the medical area with the help of this research. 


REFERENCES 

[1] Ayesha Younis 1, Li Qiang 1, *, Charles Okanda Nyatega 2,3, Mohammed Jajere Adamu 1 and Halima Bello Kawuwa.” Brain Tumor Analysis Using Deep Learning and VGG-16 Ensembling Learning Approaches”-2022

[2] Md. Sabbir Ahmed, Refeed Rahman, Shahriar Hossain, Shahnewaz Ali Mohammad D.” Brain Tumor Prediction by analyzing MRI using deep learning architectures”-2021

[3] Muhannad Faleh Alanazi 1, †, Muhammad Umair Ali 2, †, Shaik Javeed Hussain 3, *, Amad Zafar 4, *, Mohammed Mohatram 3 , Muhammad Irfan 5 , Raed AlRuwaili 1 , Mubarak Alruwaili 1 , Naif H. Ali 6” Brain Tumor/Mass Classification Framework Using Magnetic-Resonance-Imaging-Based Isolated and Developed Transfer Deep-Learning Model”-2022

[4] Muhammad Rizwan 1, (member, ieee), aysha shabbir1 , Abdul rehman Javed 2 , (member, ieee), Maryam shabbir 3 , thar baker 4 , (senior member, ieee), and dhiya al-jumeily obe 5.” Brain tumor and glioma grade classification using gaussian convolutional neural network”-2022

[5] Yogita Dubey, Mahek Narlawar, Neeraj Thosar, Dewang Bhalerao, Kajal Mitra .” Brain Tumor Classification Using Deep Learning Framework”-2022

[6] AYESHA JABBAR1, SHAHID NASEEM 1, TARIQ MAHMOOD 2,3 , TANZILA SABA 2 , (Senior Member, IEEE), FATEN S. ALAMRI 4 , AND AMJAD REHMAN 2.” Brain Tumor Detection and Multi-Grade Segmentation Through Hybrid Caps-VGGNet Model”-2023

[7] N.Siddaiah, C.Vignesh,T.Narendra,K.Pujitha,D.Madhu.” Brain Tumor detection from MRI scans using VGG- 19 networks”-2022

[8] Nadim Mahmud Dipu 1, Sifatul Alam Shohan 2 , and K. M. A Salam 3 Department of Electrical and Computer Engineering North South University, Dhaka, Bangladesh.” Brain Tumor Detection Using Various Deep Learning Algorithms”-2021

[9] Anjanayya S1, V. M. Gayathri1 and R. Pitchai2.” Brain Tumor Segmentation and Survival Prediction using Multimodal MRI Scans with Deep learning Algorithms”-2022

[10] Peicheng Wu1, Qing Chang2.” Brain Tumor Segmentation on Multimodal 3D-MRI using Deep Learning Method”-2020

[11] Somaya A. El-Feshawy , Waleed Saad , Mona Shokair, Moawad Dessouky .” Brain Tumour Classification Based on Deep Convolutional Neural Networks”-2021

[12] Dr.P. Josephin Shermila1, J. Nishitha2 and Ms. Shoba L.K.3.” Brain Tumour Detection from MRI Images Using Deep CNN”-2023

[13] Prince Priya Malla 1, *, Sudhakar Sahu 1 and Ahmed I. Alutaibi 2, *.” Classification of Tumor in Brain MR Images Using Deep Convolutional Neural Network and Global Average Pooling”-2023

[14] Marium Malik, Muhammad Arfan Jaffar, Muhammad Raza Naqvi.” Comparison of Brain Tumor Detection in MRI Images Using Straightforward Image Processing Techniques and Deep Learning 
Techniques”-2021

[15] Necip Çınar, Buket Kaya, Mehmet Kaya.” Comparison of deep learning models for brain tumor classification using MRI images”-2022

[16] Mr. C. Rajeshkumar, Dr.K. Ruba Soundar,Mrs.M.Sneha ,Ms.S.Santhana Maheswari,Ms.M.Subbu Lakshmi ,Mrs.R.Priyanka.” Convolutional Neural Networks (CNN) based Brain Tumor Detection in MRI Images”-2023

[17] Shaveta Arora,Meghna Sharma.” Deep Learning for Brain Tumor Classification from MRI Images”-2021

[18] Josmy Mathew,Dr. N Srinivasan .” Deep Convolutional Neural Network with Transfer Learning for Automatic Brain Tumor Detection from MRI”-2022

[19] A. Harshavardhan,N. Uma Maheswari ,M. Prakash ,Naresh Sammeta .” Deep Learning Algorithm for Brain Tumor Detection and Classification using MRI Images”-2023

[20] Soumya benkrama, nour el houda hemdani.” deep learning with efficientnetb1 for detecting brain tumors in mri images”-2023

[21] Anupam Pandey, Vikas Kumar Pandey.” Deep Transfer Learning Models for Brain Tumor Classification Using Magnetic Resonance Images”-2023

[22] Mrs. S. Poornam,Saravanan Alagarsamy.” Detection of Brain Tumor in MRI Images using Deep Learning Method”-2022

[23] Puneet Jain, S. Santhanalakshmi.” Early Detection of Brain Tumor and Survival Prediction Using Deep Learning and An Ensemble Learning from Radiomics Images.”-2022

[24] Samriddha Sinha, Amar Saraswat, Shweta Bansal.” Evaluation of Deep Learning Approaches for Detection of Brain Tumours using MRI”-2022

[25]Mr.P.Venkateswarlu Reddy,Dr. D.Ganesh,Ms.L.Sumaiya Azeem,Ms.M.Padma ,Ms.K.Divya Lakshmi ,Mr.L.Chakradhar.” Implementation of Latest Deep Learning Techniques for Brain Tumor Identification from MRI Images”-2023

[26] Manvi Kaur, Parul Mann,Ritu Rani, Garima Jaiswal, Arun Sharma ,Ravinder Kumar. “Implementation of Latest Deep Learning Techniques for Brain Tumor Identification from MRI Images.”-2023

[27] Arkapravo Chattopadhyay, Mausumi Maitra.” MRI-based brain tumour image detection using CNN based deep learning method”-2022

[28] Andr´es Anaya-Isaza , Leonel Mera-Jim´enez * , Lucía Verdugo-Alejo , Luis Sarasti.” Optimizing MRI-based brain tumor classification and detection using AI: A comparative analysis of neural networks, transfer learning, data augmentation, and the cross-transformer network”-2023

[29] Sumit Tripathi, Taresh Sarvesh Sharan, Shiru Sharma, Neeraj Sharma.” Segmentation of Brain Tumour in MR Images Using Modified Deep Learning Network”-2021

[30] MOHAMMAD ASHRAF OTTOM 1,2, HANIF ABDUL RAHMAN 2,3, AND IVO D. DINOV 2.” Znet: Deep Learning Approach for 2D MRI Brain Tumor Segmentation”-2022

[31] Md Ishtyaq Mahmud , Muntasir Mamun and Ahmed Abdelgawad.” A Deep Analysis of Brain Tumor Detection from MR Images Using Deep Learning Networks”-2023

[32] Francisco Javier Diaz-Pernas, Mario Martinez-Zarzuela , Miriam Anton-Rodriguez and David Gonzalez-Ortega.” A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network”-2021

[33] Kondra Pranitha, Naresh Vurukonda and Rudra Kalyan Nayak.” A Comprehensive Survey on MRI Images Classification for Brain Tumor Identification using Deep Learning Techniques”-2022

[34] Md. Shabir Khan Akash and Md. Al Mamun.” A Comparative Analysis on Predicting Brain Tumor from MRI FLAIR Images using Deep Learning”-2023

[35] Adel Kermi ,Mohamed Karam Nassim Behaz, Akram Benamar and Mohamed Tarek Khadir.”Subregions Detection and Segmentation in Multimodal Brain MRI volume”-2022

[36] Sahar Gull, Shahzad Akbar  and Ijaz Ali Shoukat.” A Deep Transfer Learning Approach for Automated Detection of Brain Tumor Through Magnetic Resonance Imaging”-2021

[37] AHMED S. MUSALLAM , AHMED S. SHERIF  AND MOHAMED K. HUSSEIN.” A New Convolutional Neural Network Architecture for Automatic Detection of Brain Tumors in Magnetic Resonance Imaging Images”-

[38] Md. Saikat Islam Khan, Anichur Rahman, Tanoy Debnath , Md. Razaul Karim , Mostofa Kamal Nasir ,Shahab S. Band , Amir Mosavi , Iman Dehzangi.” Accurate brain tumor detection using deep convolutional neural network”-2022

[39] Dr.S.Rakesh Kumar and Shashank Swaroop.” An Effective Application to Identify Brain Tumor using Deep Learning Model”-2022

[40] Shamim Ahmed, Sm Nuruzzaman Nobel, and Oli Ullah.” An Effective Deep CNN Model for Multiclass Brain Tumor Detection Using MRI Images and SHAP Explainability”-2023

[41] Annur Tasnim Islam, Sakib Mashrafi Apu, Sudipta Sarker, Syeed Alam Shuvo, Inzamam M. Hasan, Dr. Md. Ashraful Alam, and Shakib Mahmud Dipto.” An Efficient Deep Learning Approach to detect Brain Tumor Using MRI Images”-2022

[42] Adityaraj Sanjay Belhe, Vedant Vinay Ganthade, Prathamesh Suhas Uravane, Janvi Anand Pagariya and Mamoon Rashid.” An Efficient Deep Learning based Approach for the Detection of Brain Tumors”-2022

[43] Poonam Shourie, Vatsala Anand and Sheifali Gupta.” An Intelligent VGG-16 Framework for Brain Tumor Detection Using MRI-Scans”-2023

[44] Sahar Gull, Shahzad Akbar and Khadija Safdar.” An Interactive Deep Learning Approach for Brain Tumor Detection Through 3D-Magnetic Resonance Images”-2021

[45] Unnati Kosare, Prof. Leela Bitla, Sakshi Sahare, Pratiksha Dongre, Sayli Jogi and Sakshi Wasnik.” Automatic Brain Tumor Detection and Classification on MRI Images Using Deep Learning Techniques”-2023
 



