# Tensorflow Programs and Tutorials

This repository will contain Tensorflow tutorials on a lot of the most popular deep learning concepts. It'll also contain some experiments on cool papers that I read. Hopefully, the notebooks will be helpful to anyone reading!

* **CNN's with Noisy Labels** - This notebook looks at a recent [paper](https://arxiv.org/pdf/1703.08774.pdf) that discusses how convolutional neural networks that are trained on random labels (with some probability) are still able to acheive good accuracy on MNIST. I thought that the paper showed some eye-brow raising results, so I went ahead and tried it out for myself. It was pretty amazing to see that even when training a CNN with random labels 50% of the time, and the correct labels the other 50% of the time, the network was still able to get a 90+% accuracy. 

* **Character Level RNN (Work in Progress)** - This notebook shows you how to train a character level RNN in Tensorflow. The idea was inspired by Andrej Karpathy's famous [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and was based on this [Keras implementation](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/). In this notebook, you'll learn more about what the model is doing, and how you can input your own dataset, and train a model to generate similar looking text. 

* **Convolutional Neural Networks** - This notebook goes through a simple convolutional neural network implementation in Tensorflow. The model is very similar to the own described in the [Tensorflow docs](https://www.tensorflow.org/tutorials/deep_cnn). Hopefully this notebook can give you a better understanding of what is necessary to create and train your own CNNs. For a more conceptual view of CNNs, check out my introductory [blog post](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/) on them. 

* **Generative Adversarial Networks** - This notebook goes through the creation of a generative adversarial network. GANs are one of the hottest topics in deep learning. From a high level, GANs are composed of two components, a generator and a discriminator. The discriminator has the task of determining whether a given image looks natural (ie, is an image from the dataset) or looks like it has been artificially created. The task of the generator is to create natural looking images that are similar to the original data distribution, images that look natural enough to fool the discriminator network.For more of a conceptual view of GANs, check out my [blog post](https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-1-Generative-Adversarial-Nets).

* **Linear and Logistic Regression** - This notebook shows you how Tensorflow is not just a deep learning library, but is a library centered on numerical computation, which allows you to create classic machine learning models relatively easily. Linear regression and logistic regression are two of the most simple, yet useful models in all of machine learning. 

* **Simple Neural Networks** - This notebook shows you how to create simple 1 and 2 layer neural networks. We'll then see how these networks perform on MNIST, and look at the type of hyperparamters that affect a model's accuracy (network architecture, weight initialization, learning rate, etc)

* **Math in Tensorflow** - This notebook introduces you to variables, constants, and placeholders in Tensorflow. It'll go into describing sessions, and showinng you how to perform typical mathematical operations and deal with large matrices. 

* **Question Pair Classification with RNNs (Work in Progress)** - This notebook looks at the newly released question pair [dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) released by Quora a little earlier this year. It looks at the ways in which you can build a machine learning model to predict whether two sentences are duplicates of one another. Before running this notebook, it's very important to extract all the data. We'll run the following command to get our word vectors and training/testing matrices. 
   ```bash
   tar -xvzf Data/Quora/QuoraData.tar.gz
   ```

* **SELU Nonlinearity** - A recent [paper](https://arxiv.org/pdf/1706.02515.pdf) titled "Self Normalizing Neural Networks" started getting a lot of buzz starting in June 2017. The main contribution of the paper was this new nonlinear activation function called a SELU (scaled exponential linear unit). We'll be looking at how this function performs in practice with simple neural nets and CNNs. 

* **Sentiment Analysis with LSTMs** - In this notebook, we'll be looking at how to apply deep learning techniques to the task of sentiment analysis. Sentiment analysis can be thought of as the exercise of taking a sentence, paragraph, document, or any piece of natural language, and determining whether that text's emotional tone is positive, negative or neutral. We'll look at why RNNs and LSTMs are the most popular choices for handling natural language processing tasks. Be sure to run the following commands to get our word vectors and training data. 
   ```bash
   tar -xvzf Data/Sentiment/models.tar.gz
   tar -xvzf Data/Sentiment/training_data.tar.gz
   ```
   
* **Universal Approximation Theorem (Work in Progress)** - The [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that any feed forward neural network with a single hidden layer can model any function. In this notebook, I'll go through a practical example of illustrating why this theorem works, and talk about what the implications are for when you're training your own neural networks. *cough* Overfitting *cough*

* **Learning to Model the XOR Function (Work in Progress)** - XOR is one of the classic functions we see in machine learning theory textbooks. The significance is that we cannot fit a linear model to this function no matter how hard we try. In this notebook, you'll see proof of that, and you'll see how adding a simple hidden layer to the neural net can solve the problem. 
