Stable Diffusion introduced in 2022, developed by the CompViz Group at LMU Munich. 
**Idea**: Given a prompt, generate an image.

ADD DIRECTORY


### What is a generative model?
Learns a probability distribution of the data, such that we can then simple from the distribution to create instances of data. E.g., if have many pictures of cats and we train a generative model on it, we then simple from this distribution to create new images of cats.

### Why do we model data as a probability distribution?
Need for a joint distribution of the data, otherwise we may end up sampling an unreasonable image. We can also evaluate probabilities on one of the variables using conditional probability and/or by marginalizing (e.g., P(x) = Integral(P(x,y)dy)) a variable.

Each pixel is a distribution, image a joint distribution.

### Learning the distribution p(x) of our data

***insert image***

Forward process: Fixed (we define this, analytical formula)
- X0: original image
- X1: add some noise (latent variable)
- X2: add more noise
- XT: Pure noise
Reverse process: NN (we dont have analytical formula to reverse process, so we learn it)

**insert image from original paper**

Forward process q
- If I have the image at time step t-1, how can I get the more noisified version of this image:
    - We define as a Gaussian centered on the previous one and the variance defined by Beta parameter (decided by us, decide how much noise we want to add at every step, Markov chain of noisification, each variable conditioned on the previous one)
    - We also have a formula to go from the original image to any of the intermediate images. 

Reverse process pθ
- We have something noisy, we want to get something less noisy
    - We define a Gaussian mu theta, variance sigma theta, but these ones we dont know, we have to learn them, and we will use a neural network to learn them
    - We set the variance at fixed, we will parametrize it, so we only have to learn the mean

So, we have a forward process and a reverse process, and we want to learn the reverse process. How do we actually train the parameters? And our initial goal was to learn the distribution of the data, but we are learning the reverse process, so how do we get the distribution of the data?

Computing the integral is computationally itractable (we cannot marginalise). What can we do? We want to learn the parameter theta, to maximise the likelikyhood. We found a lower bound for this quantitiy, we found a lower bound for the likelihood, the ELBO. If we maximise the lower bound we will also maximise the likelihood.

Intuitive understanding: imagine you have a company:
- Revenue >= Sales
- Maximise Revenue? Maximise Sales

How practically?
**Insert pseuco code**

Idea: After getting ELBO, we can parametrise the loss function.
We need to train a network called epsilon theta, that given a noisy image at time step t and the time stamp at which the noise got added, the network as to predict has to predict how much noise is in the noisified image. And if we do gradient descent over this loss function (we minimise the loss function) we will maximise the ELBO and at the same time we will maximise the likelihood of our data.

### How do we sample from the distribution?
We start from complete noise
- we ask the network to predict how much noise is in the image
- We remove this noise
- we ask again the network to predict how much noise is in the image
- and so on

### How do we ask the model to get a picture of a cat starting from pure noise?
We introduce a signal (a prompt, context) in which we influence the model into how to remove the noise so the model will remove the noise in a way that will generate a cat.

Algorithm:
- Repeat
    - we take a sample from our dataset
    - we generate a random number t, between 1 and T
    - we sample some noise
    - We add noise to our image, and we train the model to learn to predict the amount of noise present in it
- until converged

So far, our model learns the distribution but this distribution does not know anything about the prompt (it only knows how to generate pictures similar to the training data). So idea: can we learning a joint distribution over our data and the conditioning signal, the prompt. But we dont want to learn this joint distribution (too much complexity). 

Idea: we build epsilon theta using U-net. This U-net will recieve as input a noisified image, we tell it what is the noise level of the image. The U-net has to predict how much noise is there. We also give it the prompt, this way we tell the model "can you remove the noise in this image and it is a cat". Now we have a conditioned model. So it will remove noise towards the direction of the prompt.

But at the same time, when we train the model, we can also not give any prompt at certain time.

E.g., 
STEP 1
- We give: pure noise image, noise level (T=1000), prompt(Cat)
- We get: some noise level
- We remove the noise, so we get an image towards the direction of the prompt
STEP 2
- We give: pure noise image, noise level (T=1000), NO prompt
- We get: some noise level
- We combine this output in such a way, that we can decide how much we want the output to be close to the prompt or not
    - = Classifier free guidance

output = weight * (output conditioned - output not conditioned) + output not conditioned
- weight indicates how much we want to be close to the prompt or not

### How do we give the prompt?
CLIP: Contrastive Language-Image Pre-training
Prompt: need embedded representation of the prompt, we use CLIP to get this representation

CLIP idea:
- OpenAI
- Allowed to connect text with images
- Imagine 1 is associated with Text 1, etc., build a matrix
- Matrix: dot products of the embedding of the first image with the embedding of the first text, etc.
- How train?
    - We know the correspondance between text and images is on the diagonal
    - Built a loss function for which the diagonal gets the max value

In Stable Diffusion, we take the text encoder to get the prompt embedding and this we used as conditioning signal.

### Performing many steps on big images is slow, solution? VAE
Idea: we compress the image into something smaller, we can compress this image using a VAE

image 512x512 to latent representation 64x64

Stable Diffusion is a latent diffusion model, in which we dont learn the distribution p(x) of our data set of images, but rather, the distribution of a latent representaion of our data by using a Variational Autoencoder.

Paper: High resolution image synthesis with latent diffusion models

### What is an (variational) autoencoder?
Given an image will after passing the encoder give transform into a vector with a smaller dim than original and we use this vector and run it through a decoder it will give the original back.

Problem!
Code learned by encoder is not interpretable, we dont know what it means. Code with pizza might be similar as cat.

So, we introduce the Variational Autoencoder! Now, instead of learning a code, learns a latent space. The latent space represents the parameters of a (multivariate) distribution. We learn the mean and sigma of this distribution. 

### Architecture (Text-to-image)
**Inset image**
Imagine you want to generate a pic of a dog with glasses:
- We have the prompt
- We sample noise n zero 1
- We encode the noise with VAE to get latent representation, z
- This is pure noise but compressed
- We send it to U-Net, which will predict how much noise is in the image
- We also give conditioning signal, so it predicts the noise it needs to remove so it follows a picture of a dog with glasses
- Along with time stamp
- U net will predict how much noise is in the image
- Scheduler will remove noise and send it again to U-Net for second step
- Time stamp is 980 we skip some steps
- We denoise and so on
- After T steps, no more noise
- We get latent Z' and we pass it through decoder to get image

### Architecture (Image-to-image)
Imagine you have pic of dog want to make it a dog with glasses:
- We encode image with VAE to get latent representation of image, z
- Now, we add noise to this latent representation
- **The amount of noise indicates the amount of freedom we want to give to the model to change the image!**

### Architecture (In-Painting)
We want to cut some part of the image, e.g., we cut out legs of dog and want new legs. Similar as image-to-image.
- Output of Step 1 U-net, we dont care what noise predicted for area that we want to keep
- We replace it with the original image
- We pass it again to U-net
- **So we at every step replace the area we want to keep with the original image**
- So the model itself is fooled into thinking that it is the model itself that it came up with the details

