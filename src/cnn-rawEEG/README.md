# cnn-rawEEG

This project aims to classify MI tasks based on raw EEG data (no preprocessing).

This is a rough implementation of the following article:

===========================================================================
AUTHOR=Lun Xiangmin, Yu Zhenglin, Chen Tao, Wang Fang, Hou Yimin
	 
TITLE=A Simplified CNN Classification Method for MI-EEG via the Electrode Pairs Signals  
	
JOURNAL=Frontiers in Human Neuroscience     
	
VOLUME=14      
	
YEAR=2020   
		
URL=https://www.frontiersin.org/article/10.3389/fnhum.2020.00338     
	  
DOI=10.3389/fnhum.2020.00338    
	
ISSN=1662-5161   
ABSTRACT=A brain-computer interface (BCI) based on electroencephalography (EEG) can provide independent information exchange and control channels for the brain and the outside world. However, EEG signals come from multiple electrodes, the data of which can generate multiple features. How to select electrodes and features to improve classification performance has become an urgent problem to be solved. This paper proposes a deep convolutional neural network (CNN) structure with separated temporal and spatial filters, which selects the raw EEG signals of the electrode pairs over the motor cortex region as hybrid samples without any preprocessing or artificial feature extraction operations. In the proposed structure, a 5-layer CNN has been applied to learn EEG features, a 4-layer max pooling has been used to reduce dimensionality, and a fully-connected (FC) layer has been utilized for classification. Dropout and batch normalization are also employed to reduce the risk of overfitting. In the experiment, the 4 s EEG data of 10, 20, 60, and 100 subjects from the Physionet database are used as the data source, and the motor imaginations (MI) tasks are divided into four types: left fist, right fist, both fists, and both feet. The results indicate that the global averaged accuracy on group-level classification can reach 97.28%, the area under the receiver operating characteristic (ROC) curve stands out at 0.997, and the electrode pair with the highest accuracy on 10 subjects dataset is FC3-FC4, with 98.61%. The research results also show that this CNN classification method with minimal (2) electrode can obtain high accuracy, which is an advantage over other methods on the same database. This proposed approach provides a new idea for simplifying the design of BCI systems, and accelerates the process of clinical application.

===========================================================================
