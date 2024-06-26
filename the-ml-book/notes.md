# The ML Book Notes

# Chapter 1: Introduction

The information in this section is trivial, so I am skipping it.

# Chapter 2: Notation and Definitions

## 2.1 Notation

### Data Structures
- A **scalar** is a simple numeric value: like 3 or -2.4.
- A **vector** is an ordered list of scalar values, called attributes. A vector is usually denoted by a bold character, like $\textbf{w}$ or $\textbf{x}$. The index $j$, denotes a speciic **dimension** of the vector, the position of an attribute in the list: $\textbf{w}^{(j)}$.
- A **matrix** is a rectangular array of number arranged in rows and columns. It's denoted as a capital letter - it's exactly what you think a matrix would be.
- A **set** is na unordered collection of unique elements. A set is denoted as a calligraphic capital character. A set is just a gorup of items, it can have infinite or finite size.
- **Unions** and **Intersections** of sets are easy and the reader can just look them up.

### Capital *Sigma* Notation

The summation over a collection of items $X$ or over the attributes of a vector $\textbf{x}$ is denoted as:
$$
\sum_{i=1}^{n}{\textbf{x}^{(j)}}
$$

### Capital *Pi* Notation

This is easy, instead of summing all the elements of a collection, you find the product.

### Operations on Sets

A derived set creation operator looks like this: $S' \leftarrow \{x^2 | x \in S, x > 3 \}$. This notation means that we create a new set $S'$ by putting into it $x$ squared such that $x$ is in $S$, and that $x$ is greater than 3.

The cardinality operator $|S|$ return the number of elements in the set $S$.

### Operations on Vectors

- The sum of two vectors is their element-wise sum.

- The difference of two vectors is their element-wise difference.

- A vector multiplied by a scalar is their element-wise product.

- A **dot-product** of two vectors is a scalar. Please tell me you know what the dot product is...

- A matrix times a vector is another vector.

### Functions

A **function** is a relation that associates each element $x$ of a set $X$, the **domain** of the function to a single element $y$ of another set $Y$, the **codomain** of the function. 
- You should know what a **local minimum** is.
- An **interval** is a set of real numbers with the property than any number that lies between two numbers is in also included in the set.
    - An **open interval** does not include the endpoints and is denoted using parentheses ().
    - A **closed intercal** does include the endpoints and is denoted using brackets [].
- The minimum value among all the locl minima is called the **global minimum**.
- Note: A function can return a vector $y$, even if the input is a scalar $x$, this is called a vector function.

### Max and Arg Max
Given a set of values $A$, the $max_{a \in A}{f(a)}$ returns the highest value $f(a)$ for all elements in the set $A$.

The operator $argmax_{a \in A}{f(a)}$ returns the element of the set $A$ that maximizes $f(a)$.

$min$ and $argmin$ work the same way.

### Derivative and Gradient
I will not talk too much about this section. 
- Know how to get the derivative of a function $f$.
- Know what the chain rule is.

The **gradient** is the generalization of derivative for functions that take multiple inputs. A gradient of a function is a vector of the **partial derivatives** of that function. A partial derivative can be obtained by focusing on one of the function's inputs and assuming taht all other inputs are constant values.

For example, consider a function $f([x^{(1)}, x^{(2)}]) = ax^{(1)} + bx^{(2)} + c$, then the partial derivative of the function $f$ with respect to $x^{(1)}$, denoted as $\frac{\partial f}{\partial x^{(1)}}$:

$$
\frac{\partial f}{\partial x^{(1)}} = a + 0 + 0 = a
$$

I hope this makes sense. 

The gradient of a function $f$, denoted as $\nabla f$ is given by the vector $[\frac{\partial f}{\partial x^{(1)}}, \frac{\partial f}{\partial x^{(2)}}]$.

## 2.2 Random Variable

A **random variable** usually written as an italic capital letter, like *X*, is a variable whose possible values are numerical outcomes of a random phenomenon - these variables can come from any phenomenon with a random numeric outcome like a dice roll. There are two types of random variables: **discrete** and **continuous**.

- A **discrete random variable** takes on only a countable number of distinct values such as 1, 2, 3, ..., or red, yellow, blue.
    - The probability distribution of a discrete random variable is described by a list of proababilities called a **probability mass function** (pmf): $Pr(X = heads) = 0.5$ and $Pr(X = tails) = 0.5$
- A **continuous random variable** (CRV) takes an infinite number of possible values in some interval. Because the number of values of a continuous random variable $X$ is infinite, the probability $PR(X = c)$ for any $c$ is 0. 
    - The probability distrubution of a CRV is described by a **probability density function** (pdf).

Let a discrete random variable $X$ have $k$ possible values $\{x_i\}^k_{i=1}$. The **expectation** of $X$ denoted by $E[X]$ is given by:
$$
E[X] = \sum^k_{i=1}{[x_i \cdot Pr(X = x_i)]}
$$
Where Pr($X = x_i$) is the probability that $X$ has the value $x_i$ according to the pmf. The expectation of a random variable is also called the **mean**, **average**, or **expected value** and is denoted with the letter $\mu$. 

The expectation of a continuous random variable $X$ is given by:

$$
E[X] = \int_{\mathbb{R}}{x_if_X(x)}dx
$$

where $f_x$ is the pdf of the variable X.

I hope this makes intuitive sense because this is important.

- Know **standard deviation** and **variance**.

Most of the time, we don't know the function $f$ in machine learning, but we can obser some values of $X$. These values are called **examples**, and the collections of these examples is called a **sample** or **dataset**.

## 2.3 Unbiased Estimators

Because $f_x$ is usually unknown, we can use sample $S_X = \{x_i\}^N_{i=1}$ to generate the **unbiased estimators** of the function.

We say that $\hat{\theta}(S_X)$ is an unbiased estimator of some statistic of some $\theta$ calculated using a sample $S_X$ drawn from an unknown probability distribution if $\hat{\theta}(S_X)$ has the following property:

$$
E[\hat{\theta}(S_X)] = 0
$$

where $\hat{\theta}$ is a **sample statistic**, obtained using a sample $S_X$ and not the real statistic $\theta$ that can be obtained only if $X$ is known. This means that if you can have an unlimited number of  such samples as $S_X$, and you compute an unbiased estimator such as $\hat{\mu}$ then the average of these estimators would real th real statistic $\mu$.

## 2.4 Bayes' Rule

The conditional probability Pr($X = x$ | $Y = y$) is the probability tof the random variable $X$ to have a value of $x$ *given* that another random variable $Y$ has a specific value of $y$. The **Bayes' Rule** says:

$$
Pr(X = x | Y = y) = \frac{Pr(Y=y|X=x)Pr(X = x)}{Pr(Y = y)}
$$

## 2.5 Parameter Estimation

Baye's Rule comes in handy when we have a model of $X$'s distribution and this model $f_\theta$ is a function that has some parameters in the form of a vector $\theta$. Note: this $\theta$ is not the same as the one from section 2.3, this is parameters for a function, such as the Gaussian distribution, which has two parameters, $\mu$ and $\sigma$, and is defined as:

$$
f_\theta(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

where $\sigma$ = [$\mu$, $\sigma$]. This function has all the properties of a pdf. So we can use it as a model of an unknown distribution of X. By updating the parameters of vector $\theta$ with the Bayes' Rule, we get this:

$$
Pr(\theta = \hat{\theta} | X = x) \leftarrow \frac{Pr(X = x|\theta = \hat{\theta})Pr(\theta = \hat{\theta})}{Pr(X = x)}
$$

(I don't really get this) If we have a sample $S$ of $X$ and a set of possile values for $\theta$ is finite, we can easily estimate Pr($\theta = \hat{\theta}$) by iteratively applying Bayes' Rule.

## 2.6 Parameters vs. Hyperparameters

A hyperparameter is a property of a learning algorithm, having a numeric value. This influeces how the algorithm works. Hyperparameters aren't learned by the algorithm itself form data. They have to be set by the user before running the algorithm.

Parameters are variables that define the model learned by the learning algorithm. Parameters are directly modified by the learning algorithm based on the training data. The goal is of learning is to find such values of parameters that make the model optimal.

## 2.7 Classification vs. Regression

**Classification** is a problem of automatically assigning a **label** to an **unlabelled example**. Span detection in emails is an example of classification.

In machine learning, a **classification learning algorithm** is used to learn form **labeled examples** and will produce a model that can take an unlabeled example as input and directly output a label or a value that helps determine the label such as a probability.

Is a classification problem, a label is a member of a finite set of **classes**. If the size of the classes is two (sick/healthy, spam/not spam), this is called **binary classifcation**. **Multiclass classifcation** is when there are three or more classes.

**Regression** is a problem of predicting a numeric valye label (called a **target**) given an unlabeled example. Estimating house prices based on house featuers, such as size, number of bathrooms, is an example of regression. 

The regression problem is solved by a **regression learning algorithm** that takes labeled examples as inpputs and produces a model that can take an unlabeled example as an input and output a target. The target for a regression problem does not have to be in a finite set.

## 2.8 Model-Based vs. Instance-Based Learning

Most supervised learning algorithms are model-based. An SVM is model based, and they use training data to create a **model** that has **parameters** that it learned form the trianing data. (this one seems to be smarter)

Instance-based learning algorithms use the whole dataset as the model. The **k-Nearest Neighbors (kNN)** model is an instance-based learning algorithm. The predict a label for an input example, the kNN algorithm looks at the close neighborhood of the input example and outputs the label that it saw the most often. (this one seems to be dumber)

## 2.9 Shallow vs. Deep Learning

A **shallow learning** algorithm learns the parameters of the model directly from the featurs of the trianing examples. Most supervised learning algorithms are shallow. 

An exception is the **neural network**, especially those with more than one hidden **layer** between input and output. These are called **deep nueral networks**. In **deep learning** models must tune parameters not directly from the features of the trianing examples, but from the outputs of proceeding layers. (it's okay to not know this yet)

# Chapter 3: Fundamental Algorithms

## 3.1 Linear Regression

**Linear regression** is a popular regression algorithm that learns a model which is a linear combination of features of the input example.

### Problem Statement

We have a collection of labeled examples, $\{(x_i, y_i)\}^N_{i=1}$, where $N$ is the size of the collection, $x_i$ is the D-dimensional feature vector, and $y_i$ is a real-valued target.

We want to build a model $f_{w, b}(x) = wx + b$. We want to find optimal values $(w^*, b^*)$. The hyperplane in linear regression is chose to be as close to all training examples as possble.

### Solution

To get this latter requirement satisfied, the optimization process includes minimizing the following expression:

$$
\frac{1}{N}\sum_{i = i...N}(f_{w,b}(x_i) - y_i)^2
$$

The expression we minimize or maximize is called the **objective function** or an **objective**. The expression $(f_{w,b}(x_i) - y_i)^2$ is called the **loss function**, this specific one is called the **squared error loss**. For machine learning we try to minimize the objective, which is known as the **cost function**. In this example, the cost function is the **empirical risk**, or the average loss.

Different loss functions can work. New loss functions would result in new and innovative model architectures. But, just because it could work in practice, doesn't mean it will perform better in practice.

There are two reasons new models are invented:

1. The new algorithm solves a specific practical problem better than the existing algorithms. 
2. The new algorithm has better theoretical garuantees on the quality of the model is produces. 

Why use a complicated model when you can use a simple one? Simple models rarely overfit. **Overfitting** is the property of a model such that it performs very well on training examples, but does not perform well on example that haven't been seen by the learning algorithm.

We use the squared difference for two reasons, 1) the loss function is a smooth, differentiable graph, compared to the absolute differences, and 2) it exaggerates larger differences between two points. 

## 3.2 Logistic Regression

Logistic regression is not a regression algorithm, it's a classification algorithm.

### Problem Statement

In **logistic regression**, we still want to model $y_i$ as a linear function fo $x_i$, but with a binary $y_i$. This is hard. If we define a negative label as 0 and the positive label as 1, we just need a continuous function with the codomain [0, 1]. If the value returned by the model for a input is closer to 0, then the negative label is assigned to x. A function we can use the is **standard logistic function**, or the **sigmoid**.
$$
f(x) = \frac{1}{1+x^{-x}}
$$

The logistic regression model looks like this:
$$
f_{x,b}(x) = \frac{1}{1+e^{-(wx+b)}}
$$

How do we find optimal $w, b$? In linear regression we minimized the empirical risk, which was the average squared error loss, which is also known as the **mean sqaured error (MSE)**.

### Solution

In logistic regression, we maximize the **likelihood** of our training set according to the model. In statistics, the likelihood function defines how likely an example is according to the model.

For example, we have a labeled example $(x_i, y_i)$ in our training data. We also has some random values for our parameters $w, b$. If we apply the logistic regression model, we get some value $0 < p < 1$ as output. If $y_i$ is the positive class, the likelihood of $y_i$ being the positive class is given by the value $p$.

The optimization function in logistic regression is called **maximum likelihood**. We maximize the likelihood of the training data according to our model:

$$
L_{w, b} = \prod_{i=1...N}f_{w, b}(x_i)^{y_i} (1-f_{w, b}(x_i))^{(1-y_i)}
$$

This $f_{w, b}(x_i)^{y_i} (1-f_{w, b}(x_i))^{(1-y_i)}$ just means, $f_{w, b}(x_i)^{y_i}$ if $y_i = 1$ and $(1-f_{w, b}(x_i))$ if $y_i = 0$. Just plug it in for the zero in the exponent.

We use the product operator instead of the sum operator because the likelihood of observing N labels for N examples is the product of the likelihood of each observation, it's like probabilities.

To avoid **numeric overflow**, it more convenient to use the **log-likelihood**. The log-likelihood is defined as follows:

$$
LogL_{w, b} = \ln(L_{w, b}(x)) = \sum_{i=1}^{N}[y_i\ln f_{w, b}(x) + (1-y_i)\ln (1-f_{w, b}(x))]
$$

Since the natural log is a strictly increasing fuction, maximizing this function is the same as maximizing the argument that is passed into the function. 

## 3.3 Decision Tree Learning

A **decision tree** is an acyclic **graph** that can be used to make decisions. In each branching node, a specific feature $j$ of the feature vector is used. If the feature is below a threshold, the left branch is followed, otherwise the right branch is followed. If a leaf node is reached, the decision is made as to what class the example belongs. 

### Problem Statement

Similarly, we have a collection of labeled examples: labels are all in the set ${0,1}$. We want to build a decision tree that helps us predict the class a give feature belongs in.

### Solution

We will focus on the ID3 decision tree in this section. The optimization criterion is the average log-likelihood.

Contrary to logistic regression which builds a **parametric model** $f_{w, b}$ by finding an optimized set parameters to the optimization criterion. The ID3 algorithm optimizes it by constructing a **nonparametric model**.

This is how it works. Let $S$ denote a set of labeled examples. I the beginning, the decision tree only has a start node that contains all examples. Sart with a constant model $f_{ID3}^{S} = \frac{1}{|S|}\sum_{(x, y) \in S} y$.

The prediction would be the same for any input $x$. We then search through all features and all thresholds, and split the set $S$ into two subsets: $S_-, S_+$, the first where all features are lower than the threshold, and the second where all features are higher than the threshold. Finally we pick the best values $(j, t)$ to form two new leaf nodes, and recursively run the algorithm.

To evaluate how "good" a split is we evaluate the **entropy**. Entropy is a measure of uncertainty about a random variable. The entropy of a set is given by:

$$
H(S) = -f^S_{ID3}\ln f^{S}_{ID3} - (1-f^S_{ID3})\ln (1-f^S_{ID3})
$$

The entropy of a split is $H(S_-, S_+) = \frac{|S_-|}{|S|}H(S_-) + \frac{|S_+|}{|S|}H(S_+)$. So at each step, at each leaf node, we find a split that minimizes the entropy. 

The algorithm stops when:
- All examples in the lead node are classified correctly by the one-piece model.
- We cannot find an attribute to split upon.
- The split reduces the entropy by less than some threshold.
- The tree reaches some maximum depth.

## 3.4 Support Vector Machine

This has been introduced, so we are just filling in the blanks:

1. What is there's noise in the data and it's almost impossible to perfectly fit a hyperplane?
2. What is the data cannot be separated using a plane, but only using a higher order polynomial?

We want to remember the following constraints:

$$
wx_i - b \ge +1 \text{ if } y_i = +1 \\
wx_i - b \le -1 \text{ if } y_i = -1
$$

We also want to minimize $||w||$ so that the hyperplane is equally distant from the closest examples f each class. To minimize $||w||$, we can minimize $\frac{1}{2}||w||^2$. So the optimization problem for an SVM is:

$$
\text{min}\frac{1}{2}||w||^2, \text{ such that } y_i(wx_i - b) \ge 0, i = 1, ..., N.
$$

### Dealing with Noise

To deal with noise, we introduce the **hinge loss** function: $$\text{max}(0, 1-y_i(wx_i-b))$.

If these constraints are met:

$$
wx_i - b \ge +1 \text{ if } y_i = +1 \\
wx_i - b \le -1 \text{ if } y_i = -1
$$

If $wx_i$ is on the correct side of the decision boundary, the hinge loss is zero. For data on the wrong side of the boundary, the function's valye is proportional to the distance from the decision boundary. The cost function becomes:

$$
C||w||^2 + \frac{1}{N}\sum^N_{i=1}\text{max}(0, 1-y_i(wx_i - b))
$$

Where the hyperparameter C determines the tradeoff between increasing the size of the decisio boundary and ensuring that each $x_i$ lies on the correct side of the decision boundary. This value can be found experimentally. 

SVMs with optimize hinge-loss are called sot margin SVMs, while the original formulation is a hard-margin SVM.

As the value of C increases, the second term in the cost function will lose significance. The SVM model tries to find the highest margin by completely ignoring misclassifciation. As C becomes smaller, making classification errors is more costly, and the model tries to minimize the width of the margin to make less mistakes.

### Dealing with Inherernt Non-Linearity

SVMs can also be used on datasets that cannot be separated by a hyperplane in it's original space. But, we can change the space to make the data linearly separable. Using a function to implicitly transform the original space into a higher dimensional space is called the **kernel trick**.

The idea of the kernel trick is to transform two-dimensional data into a linearly separatale n-dimensional data using a specific mapping. However, we don't know what mapping would work for our data.

BUT! we've figured out how to use **kernel functions** to efficiently work in higher dimensional spaces without doing the transformation explicitly - *that's crazy*. Let's see how this works, but we first need to see how the optiization algorithm for SVM finds the optimal values for $w$ and $b$.

You can skip the next few paragraphs if you're weak. The method traditionally used to solve this problem is the method of *Lagrange Multipliers*. The original problem is the same as:

$$
\max_{\alpha_1...\alpha_n}\sum_{i=1}^N\alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{k=1}^{N}y_i\alpha_i(x_ix_k)y_k\alpha_k
$$

where $\alpha_i$ are called Lagrange multipliers. THe optimization becomes a quadratic optimization problem - which is easy for computers to do. Please read into Lagrange multipliers if you are interested.

Multiple kernel functions exist, most famous of which is the **RBF kernel**:

$$
k(x, x') = \exp(\frac{||x-x'||^2}{2\sigma^2})
$$

where $||x-x'||^2$ is the squared **Euclidean distance** between two features vectors. The feature space fo the RBF kernel has an infinite number of dimensions. By changing the hyperprameter *sigma*, the scientist can choose between getting a smooth or curvy decision boundary.

## 3.5 K-Nearest Neighbors

**k-Nearest Neighbors (kNN)** is a non-parametric learning algorithm. kNN keeps all the training data in memory, it needs it. Once a new, previously unseen examples $x$ comes in, the kNN algorithm finds $k$ training examples closest to $x$ and returns the majority or average label.

The closeness of two examples is given by a distance function. THe euclidean distance is frequently used, another distance is the **cosine similarity**:

$$
s(x_i, x_k) = \cos((x_i, x_k)) = \frac{\sum^D_{j=1}x_i^{(j)}x_k^{(j)}}{\sqrt{\sum_{j=1}^D(x_i^{()j})^2\sum_{j=1}^D(x_k^{()j})^2}}
$$

This is the measure of the similarity of the directions of the two vectors. If the two vectors were in the same direction, then the two vectors point in the same direction, and the cosine similarity is 1. Think of it as dot product, over the product of the magnitudes of the vector, you get just the cosine of the angle. Other distances include the Chebychev distance, Mahalanobis distance, and the Hamming distance.

# Chapter 4: Anatomy of a Learning Algorithm

## 4.1 Building Blocks of a Learning Algorithm

Each learning algorithm consists of three parts:

1) a loss function
2) an optimization criterion based on the loss function
3) an optimization routine leveraging training data to find a solution that optimizes the criterion

**gradient descent** and **stochastic gradient descent** are two most frequently used algorithms used in cases where the optimization criterion is differentiable.

Gradient descent is a tool for finding the minimum of a function. Gradient descent can be used to find optimal parameters.

## 4.2 Gradient Descent

Welcome to multivariable calculus at the Peddie School! The date is January 12th, 2023. The optimization criterion will include two parameters, $w$ and $b$. The dataset that we will inspect is the relationship between the spending of a radio compay, and the sales of the company, in units sold. 

Let's say we have 200 example pairs. The linear regression model is $f(x) = wx + b$, but we don't know what the optimal values are for the parameters. To do that, we want to find values for $w$ and $b$ that minimizes the sqaured error.

$$
l = \frac{1}{N}\sum^N_{i=1}(y_i-(wx_i+b))^2
$$

We need to start by calculating the partial derivative for every parameter:

$$
\frac{\partial l}{\partial w} = \frac{1}{N}\sum^N_{i=1}-2x_i(y_i-(wx_i+b))\\

\frac{\partial l}{\partial b} = \frac{1}{N}\sum^N_{i=1}-2(y_i-(wx_i))
$$

Gradient descent proceeds in **epochs**, an epoch is using the training set entirely to update each parameter. In the first epoch, we initialize $w = 0$ and $b = 0$. At each epoch, we update $w$ and $b$ using their partial derivatives. The **learning rate** $\alpha$ controls the size of an update:
$$
w = w - \alpha\frac{\partial l}{\partial w} \\
\text{}\\
b = b - \alpha\frac{\partial l}{\partial b}
$$

We subtract a fraction of the fartial derivatives because we want to go in the opposite directions the derivatives point in. The derivatives point in directions of growth, and we want to go the opposite way. 

At the next epoch, we recalculate the partial derivatives with the new values of $w$ and $b$. We continue this process until we see that $w$ and $b$ stop changing as much, then we stop.

Gradient descent is slow and is very sensitive to the choice of the learning rate. It's good that we are smart and imrpovements have been made.

**Minibatch stochastic gradient descent** (minibatch SGD) speeds up the computation by approximating the gradient using smaller batches of the training data. SGD has many internal upgrades as well. **Adagrad** is a version of SGD that scaled the learning rate $\alpha$ for each parameter according to the history of gradients. $\alpha$ is reduced for very big gradients and vice-versa. **Momentum** is a method that helps accelerate SGD by orienting the gradient descent in the relevant direction and reducing oscillations. In neural network training, variants of SGD such as **RMSprop** and **Adam** are used. 

These are not machine learning algorithms, but solvers of minimization problems where the function to minimize has a gradient.

## 4.3 How Machine Learning Engineers Work

Unless you are a cool person, you shouldn't write your own machine learning algorithms, but you should know how they work, that's why you're reading this book. 

You use libraries - hopefully you know what that is - most of which are open source. Libraries that are commonly used include PyTorch, Scikit-learn, and Tensorflow.

These libraries make things really, really easy.

## 4.4 Learning Algorithms' Particularities

