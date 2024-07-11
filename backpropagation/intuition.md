

Let's start with a very simple network: 4 layers, where each layer only has one neuron in it. There are three weights and three biases that determine the value of the cost function.

Intuitively, we want to find which weights and biases affects the cost function the most. Changing that will help us find the minimum of the cost function. 

Let's focus on the last two layers. We denote the activation of the last layer as $a^{(L)}$, and the activation of the layer before that is $a^{(L-1)}$. The superscript denotes which layer the neuron is in. The desired output for a training example is denoted as $y$.

The cost function, at the end of the network, would be:
$$
C_0 =  (y - a^{(L)})^2
$$

This last activation, is calculated by a weight, a bias, and the previous activation all going into an activatin function $\sigma$:
$$
a^{(L)} = \sigma(w^{(L)}a^{(L-1)}+b^{(L)})
$$

It'l be helpful to give this weighted sum a definition, so $z^{(L)} = w^{(L)}a^{(L-1)}+b^{(L)}$ and we get:
$$
a^{(L)} = \sigma(z^{(L)})
$$

To mentally visualize all these mental variables, for a given laye $L$, a weight ($w^{(L)}$), a bias ($b^{(L)}$), and the acviation of the previous layer ($a^{(L-1)}$), allow us to calculate the weighted sum $z^{(L)}$. We can then use an activation function like the sigmoid or ReLU, along with $z^{(L)}$ to calculate $a^{(L)}$. And the squared difference between $a^{(L)}$ and $y$ is our cost function.

Of course, $a^{(L-1)}$ has it's own weights and biases that it's calculated from.

Now focusing on just the parameters of the last two layers, what we want is to find how sensitive the cost function is to a change in the weight and bias of the last layer by examining the partial derivatives of the cost function to those respective variables:

$$
\frac{\partial C_0}{\partial w^{(L)}} \text{ and } \frac{\partial C_0}{\partial b^{(L)}}
$$

Since we know that $C_0$ is a function of $a^{(L)}$ which is a function of $z^{(L)}$ which is a function of $w^{(L)}$, we can use the chain rule:
$$
\frac{\partial C_0}{\partial w^{(L)}} = \frac{\partial z^{(L)}}{\partial w^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}} \frac{\partial C_0}{\partial a^{(L)}}  
$$

Let's take this derivative by derivative:

$\frac{\partial C_0}{\partial a^{(L)}}$ doesn't look too bad, remember $C_0 = (a^{(L)}-y)^2$. So $\frac{\partial C_0}{\partial a^{(L)}} = 2(a^{(L)} - y)$ due to the power rule.


$\frac{\partial a^{(L)}}{\partial z^{(L)}}$ is just wrapped through an activation function: ReLU or sigmoid, $a^{(L)} = \sigma (z^{(L)})$. $\frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma ' (z^{(L)})$ by the definition of a derivative.

$\frac{\partial z^{(L)}}{\partial w^{(L)}}$ is the simplest one, since $z^{(L)} = w^{(L)}a^{(L-1)} + b^{(L)}$. So the derivative, $\frac{\partial z^{(L)}}{\partial w^{(L)}}$ is simply $a^{(L-1)}$.

This is alot, so let's explain a few things. For the last derivative, the amount that the change of the weight in the layer $L$, depends on how strong the activation of the neuron in the previous layer $L-1$ is.

This is just the derivative with respect to $w^{(L)}$ of just the cost function of one training example. The full cost requires the averaging of the cost of all training examples:
$$
\frac{\partial C}{\partial w^{(L)}} = \frac{1}{N}\sum^{n-1}_{k=0}\frac{\partial C_k}{\partial w^{(L)}}
$$

And of course, that partial derivative, is itself, just one part of the gradient of the cost function, and the rest of the gradient is made up of partial derivatives with respect to the rest of the weights and biases. 

Now, we see that we can find how much the weighted sum $z^{(L)}$ shifts with respect to the activation of the previous layer, $a^{(L-1)}$, which is the weight of the layer $L$, which makes should sense.

So let's generalize: most networks in practice are not this simple. They have multiple layers each with multiple nuerons. It's actually not that complex, so let's break it down.

Rather than the activation of a layer being just $a^{(L)}$, it is now going to have a subscript, indicating which neuron it's from, such as $a^{(L)}_0$. Let's use the letter $j$ to index the activations of the layer $L$ and the letter $k$ to index the activations of the layer $L-1$. Now the cost function becomes:
$$
C_0 = \sum^{n_L - 1}_{j=1}(w^{(L)}_j - y_j)^2 
$$

Since there are alot more weights and alot more biases let's denote the weight that connects the $k$th neuron in layer $L-1$ to the $j$th neuron in layer $L$, as $w^{(L)}_{jk}$. Let's do the same thing and name this weighted sum $z^{(L)}_j = w^{(L)}_{j0}a^{(L-1)}_0+w^{(L)}_{j1}a^{(L-1)}_1 + w^{(L)}_{j2}a^{(L-1)}_2 + b^{(L)}$.

And to calculate the activation of layer $L$, we just wrap the weighted sum in a activation function:
$$
a^{(L)}_j = \sigma(z^{(L)}_j)
$$

This should look very similar, and is also why the partial derivative of the cost function with respect to the weight in layer $L$ at neuron $j$ is:
$$
\frac{\partial C_0}{\partial w^{(L)}_{jk}} = \frac{\partial z^{(L)}_{j}}{\partial w^{(L)}_{jk}} \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j} \frac{\partial C_0}{\partial a^{(L)}_j}  
$$

What is different though, is the derivative of the cost with respect to the activations of a previous layer $L-1$. This neuron actuall influences the cost through multiple paths - more specifically, each neuron in layer $L$. Therefore, we just take the sum of the derivatives of each neuron in layer $L$:
$$
\frac{\partial C_0}{\partial a^{(L-1)}_k} = \sum_{j=0}^{n_L-1}\frac{\partial z^{(L)}_{j}}{\partial a^{(L-1)}_k} \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j} \frac{\partial C_0}{\partial a^{(L)}_j}  
$$

This process for calculating the derivative of the cost function with respect to a weight from the second to last layer can be repeated for virtually any other layer.


