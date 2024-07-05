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
f(x) = \frac{1}{1+e^{-x}}
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

Some algorithms, like decision tree learning, can accept categorical features. If there is a feature "color" that can take values "red", "yellow, or "green" you can keep this feature as is. SVMs, logistic and linear regression, and kNN expect numerical values for all feature. 

Some algorithms, like SVMs, allow the user to provide weightings for each cass. These weightings influence how the decision boundary is drawn. If the weight of some class is high, the learning algorithm will try to not make mistakes when classifying based off of that feture. 

Some classification algorithms build the model using the whole dataset at once (decision tree learning, logistic regression, or SVMs). If you have more data, you would have to rebuild the model from scrtch. Other algorithms (Naive Bayes, multilayer perceptron, SGDClassifier/SGDRegressor) can be trained iteratively, one batch at a time. If you have new training data, you can update the model with that data. 

Some algorithms can be used for both classification and regression, such as decision tree learning, SVMs, and kNNs, but not both. 

# Chapter 5: Basic Practice

There are some issues we need to fix before we implement any of these libraries that make our lives easy. Before that, our lives are hard.

## 5.1 Feature Engineering

Before doing any machine learning project, you need the **dataset** first. The dataset is the collection of **labeled examples** with $N$ **feature vectors**. A feature vector is a $j$ dimensional vector that contains a value that describes the example somehow. That value is a **feature** and is denoted as $x^{(j)}$.

The problem of transforming raw data into a dataset is called **feature engineering**. For example, to transform the logs of user interaction with a computer system, one could create features that contain information about the user and various statistics extracted from the logs. Everything measurabel can be used as a feature. 

You want to create *informative* features. The better a feature is, the higher its *predictive power*. A model has a **low bias** when it predicts the training data well.

### One-Hot Encoding

Most learning algorithms only work with numerical feature vectors. If you have categorical features like, "colors" or "day of the week", those can be transformed into binary ones.

If your data has three possible feature "colors" and they are labeled "red", "yellow", and "green", you can transform them into three vectors:

$$
\text{red} = [1, 0, 0] \\
\text{yellow} = [0, 1, 0] \\
\text{gree} = [0, 0, 1] \\
$$

You increase the dimensionality of the vectors - you shouldn't change them by changing red into 1, yellow into 2, and green into 3, that would imply an order to the features. If the order of a feature's values is not important, using ordered numbers as values is likely to confuse the learning algorithm as it tries to find regularity where there's no one correct answer which can lead to possible overfitting.

### Binning

When you have a numerical feature, but want to convert it to a cateforical one, you can use **binning**. Binning is the process of converting a continuous feature into multiple binary features called bins or buckets, usually based on value range. For age, bins could be from 1-10 years, 11-20 years, etc.. 

Let feature $j = 4$ represent age. By applying binning, we replace this feature with the corresponding bins. Let the three new bins, be added with indices $j = 123$, $j = 124$, $j = 125$ respectively. Now if $x_i^{(4)} = 7$ for some example $x_i$, then we set feature $x_i^{(124)}$ to 1, if $x_i^{(4)} = 13$, thenw e set feature $x_i^{(125)}$ = 1.

In some cases, a carefully desgiend binning can help the learning algorithm use fewer examples, because you are already limiting uncertainty.

### Normalization

**Normalization** is the process of converting an actual range of values which a numerical feature can take, into a standard range of values, typically in the interval $[-1, 1]$ or $[0, 1]$.

For example, supposed the natural range of a particular feature is 350 to 1450. But subtracting 350 from every feature and dividing it by 1450 - 350 = 1100, one can normalize those values into the range $[0, 1]$:

$$
\bar{x^{(j)}} = \frac{x^{(j)} - min^{(j)}}{max^{(j)} - min^{(j)}}
$$

You don't have to normalize your data. But it can help increase the speed of learning. It would improve the scale of the gradient during gradient descent, especially if the features are not on the same scale. If one feature is in the range $[0, 1000]$ and the other is in $[0, 0.01]$, then the derivative with respect to a larger input feature would dominate the learning algorithm. This also helps to avoid **numeric overflow** where numbers get too big for computers to handle.

### Standardization

**Standardization** or **z-score normalization** is a rescaling process so that they have the properties of a standard normal distribution with $\mu = 0$ and $\sigma = 1$ where $\mu$ is the mean of the dataset and $\sigma$ is the stardard deviation from the mean.

Standard scores of features are calculated as follows:

$$
\hat{x^{(j)}} = \frac{x^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
$$

There's no definitive answer as to whether you should use normalization or standardization. Unsupervised learning algorithms usually perform faster from standardization. Standardization is also preferred if the data already takes a normal distribution. Standardization is also preferred if features can have extremely high or extremely low values, because normalized values will squeeze the values into a tight range.

Every other time, use normalization.

### Dealing with Missing Features

Sometimes, you'll have data in the form of a dataset that already has features defined. Sometimes, values for some features can be missing or incomplete. This is what you can do:

- Remove the incomplete examples from your dataset - if your dataset is large enough
- Using a learning algorithm that can deal with missing feature values 
- Use a **data imputation** technique

### Data Imputation Techniques

One data imputation technique consists in replacing the missing value of a feature by an average value of this feature in the dataset.

Another technique is to replace the missing value with a value outside the normal range of values. FOr example, if the normal range is $[0,1]$ then you can set the missing value to 2. The idea is that the learning algorithm will learn what to do when the feature has a value that is very different from regular values.

Or you can replace the value with a value that is the in the middle of the range, the idea is that the value in the middle will not signiicantly affect the prediction.

A more advanced technique is the use the missing value as a target variable as a regression problem. You can use the remaining features to form the feature vector, and predict the missing value. 

Lastly, if you have a large enough dataset, you can add a dimension to each feature vector, adding a binary indicator feature for each feature with the missing value. 

You should always use data imputation to clean up your data. Try several techniques, some might work better than others.

## 5.2 Learning Algorithm Selection

Choosing a machine learning algorithm can be hard, arguably the hardest task in this process. You can ask questions about your problem and about your data before starting to work on the problem.

- Explainability: Does you model have to be explainable to a non-technical audience? Most very accurate algorithms are black boxes. They learn models that make very very few errors, but why a model makes a specific prediction could be very hard to understand and even harder to explain. Nueral networks often have this issue.

    kNNs, linear regression, or decision tree learning produce models that aren't super accurate, but are easily explained.

- In-memory vs. out-of-memory: Can your dataset be fully loaded into the RAM of your system? If yes, then you can choose almost all algorithms. Else, you would prefer **incremental learning algorithms** that improves the learning algorithm gradually with data. 

- Number of features and examples: How many training examples do you have? How many feature does each example have? **Neural networks** and **gradient boosting** can handle a huge number of examples and millions of features. SVMs on the other hand, can't do that.

- Categorical vs. numerical features: Does your data have categorical or numerical features? or both? Depending on your answer, you need different algorithms.

- Nonlinearity of the data: Is your data linearly separable or can it be modeled using a linear model? If yes, SVM with the linear kernel, logistic or linear regression can be good choices. Or else you're going to need a deep neural network.

- Training speed: How much time do you have for training the model, you're going to make mistakes, so make sure you know. Neural networks are going to be slow. Simpler algorithms are going to be faster. 

- Prediciton speed: How fast does the model have to be whne generating predictions? Is it going to be in high production? SVMs, regression, and some neural networks are extremely fast, whereas kNN, ensemble algorithms, and very very deep neural netowrks can take more time.

## 5.3 Three Sets

In practice, users work with three distinct sets of labeled examples:

1. training set
2. validation set
3. test set

Once you have your annotated dataset, the first thing you do is shuffle your examples and split them into three subsets: **training, validation** and **test**. The training set should be the biggest one, the one you use to build the model. The validation and test sets should be around the same size, much smaller than the training set. The model should not see the two latter sets during learning. These sets are called **holdout sets**. 

When we build a model, what we don't want is for themodel to only do well at predicting the examples that the learning algorithm has already seen. Assume there's a learning algorithm that just memorizes all training examples and then uses memory to predict their labels. This algorithm will make no mistakes during training, but would be useless in the practical world.

What you want is a model that is good at predicting exampels that the learning algorithm didn't see: you want high performance on holdout sets.

What's the difference between the validation set and the test set? We use the validation set to 1) choose the learning algorithm and 2) find the best values of the hyperparameters. The test set is used to asses the model before putting it in production.

## 5.4 Underfitting and Overfitting

A perfect model would have no **bias**. A model with low bias means that it predicts the labels of the training data well. If the model makes any mistakes on the training data, it has a **high bias** and the model **underfits**. Underfitting is the inability of the model to predict the labels of the data it was trained on well. There are two important reasons for this:
- your model is too simple of the data
- your features are not informative enough

For the first problem, if your data resembles a curved line, but your model is a straight line, it's simplicity holds it back from being able to learn the extra complexity present in the data. For the second problem, take the example of predicting cancer, but all you have is the height and weight of an individual - not super helpful.

To solve underfitting - find a more complex model, or get more informative features.

**Overfitting** on the other hand, is when the model predicts very well on the training data, but poorly on the holdout sets. Two important factors that can lead to overfitting is:
- your model is too complex for the data
- there are two many features but not many training examples

Overfitting is also called **high variance**. This means that is your training data was sampled differently, your model would produce a dramatically different model. 

Even a simple model, like a linear model, can overfit the data. This happens when the data is high-dimensional and there are not alot of training examples. When feature vectors are high-dimensional, linear learning algorithms can assign non-zero value to all parameters in the parameter vector $w$. It's trying too hard to find complex relationships between all available features to predict the labels perfectly. 

A complex model on the other hand, also has a high risk of overfitting. It tries to percfectly predict the labels of all training examples, and will learn the specifics of each set: the noise, the sampling, and other features that don't really matter. 

To solve overfitting - try a simpler model, reduce the dimensionality of your data, add more training data examples, or *regularize* the model.

## 5.5 Regularization

**Regularization** is an umbrella term that includes methods that force the learning algorithm to build a less complex model. This leads to higher bias but less variance, this is the **bias-variance tradeoff**.

The most common regularization are called **L1** and **L2 regularization**. To create a regularized model, we change the objective function by adding a penalizing term that is high when the model is complex. Let's use linear regression for an example:
$$
\min_{w, b}\frac{1}{N}\sum^N_{i=1}(f_{w, b}(x_i) - y_i)^2
$$

An L1 regularized objective function looks like this:

$$
\min_{w, b}[C|w| + \frac{1}{N}\sum^N_{i=1}(f_{w, b}(x_i) - y_i)^2]
$$

where $|w| = \sum^D_{j=1}|w^{(j)}|$ and $C$ is a hyperparameter. The higher the value for $C$ is, the learning algoithm will try to set most $w^{(j)}$ to a small value or zero to minimize the objective. But this can lead to underfitting, so the task is to find the highest $C$ value that will not lead to underfitting.

An L2 regularized objective looks like this:
$$
\min_{w, b}[C||w||^2 + \frac{1}{N}\sum^N_{i=1}(f_{w, b}(x_i) - y_i)^2], \text{ where } ||w||^2 = \sum^D_{j = 1}(w^{(j)})^2
$$

In practice, L1 regularization produces a **sparse model**, where most of the parameters are close to zero. L1 performs **feature selection** by deciding which features are the most important for prediction. However, L2 maximizes the performance of the model on holdout data. It has the advantage of being differentiable, so gradient descent can be used to optimize the objective function.

L1 and L2 regularization is also combined in **elastic net regularization**. L2 is called **ridge regularization** and L1 is called **lasso regularization**.

Two other common techniques for regularization for neural networks are **dropout** and **batch normalization**. Non-mathematical methods also include **data augmentation** and **early stopping**.

## 5.6 Model Performance

How do you tell how well a model performs? You use the test set to assess the model. If the model performs well on the test set, data it's never seen before, we say that it **generalizes well**.

Formal metrics and tools are used to assess the model performcance. For regression, a well-fitting regression model reuslts in predicted values close to the observed values. A **mean model** which always predicts the average of the labels in the training data, is a baseline to compare to. Good models will outperform the mean model.

Then, we can look at the mean squared error of the data. If the MSE of the model on the test set is significantly higher that the MSE obtained on the trianing data, this is a sign of overfitting. Obviously, MSE isn't the only way to tell how well a model performs.

For classification here are some of the things we can do:
- confusion matrix
- accuracy
- cost-sensitive accuracy
- precision & recall & area under ROC curve

### Confusion Matrix

The **confusion matrix** is a table that summarizes how successful a classifcation model is at predicting examples. The two axes of the confusion matrix is the label that the model predicted, and the actual label of the data. Let's take categorizing spam emails as an email. The confusion matrix counts **true positives**, the number of spam emails classified as spam emails, **false negatives**, the number of spam emails classified as non-spam emails (incorrectly predicting the negative class), **true negatives** the number of non-spam emails classified as non-spam emails, and **false positives** non-spam emails classified as spam emails (incorrectly predicting the positive class).

The confusion matrix is used to calculate **precision** and **recall**.

### Precision & Recall

**Precision** and **recall** are the two most commonly used metrics to assess model performance when applicable. Precision is the ratio of correct positive predicitons to the overall number of positive predictions:
$$
\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$
Recall is the ratio of correct positive predictions to the overall number of positive examples in the dataset:
$$
\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Let's take a document retrieval task with a query. The precision is the proportion of relevant documents in the list of all returned documents by the model. The recall si the ratio of the relevant documents returned by the model to the total number of relevant documents that exists - regardless of whether or not they were retrieved.

For spam detection, we want to have high precision, and not miscategorize a non-spam email as a spam email, and can tolerate low recall, because it's fine if some spam makes it into the inbox. It's usually imposible to have both.

To extend this to multiclass prediction, you can only calculate precision and recall on one class, and assume the other classes to be negatives.

### Accuracy

**Accuracy is the number of correctly classified examples (TP & TN), divided by the total number of classified examples:
$$
\text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FN} + \text{FP}}
$$

Accuracy is good when you want to be able to predict all classes well.

### Cost-Sensitive Accuracy

When classes have different importance, we can use **cost-sensitive accuracy**. To compute this, assign a cost (positive real number) to both types of mistakes: false positives, and false negatives. When calculating the new accuracy, multiply the FPs and FNs by that coefficient cost.

### Area under the ROC Curve (AUC)

The ROC Curve is the "reciever operating characteristic" curve and is used to assess the performance of classification models. The ROC curves uses the **true positive rate**, which is given in recall, and the **false positive rate** (proportional of negative examples predicted incorrectly).
$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \text{ and } \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

ROC Curves can only be used to assess classifiers that return some confidence score for prediction. To draw the curve, you first discretize the range of confidence scores: [0, 0.1, 0.2, 0.3, 0.4, 0.5, ...]. Then you can use each discrete value as the prediction threshold and predict the labels of examples in your dataset using the model and this threshold. Example: you want to compute TPR and FPR for the threshold equal to 0.7, you apply the model to each example, get the score, and if the score is higher than 0.7 you predict the positive class.

The higher the **area under the ROC curve**, the better the classifier. A classifier with an AUC higher than 0.5 is better than a random classifier.

## 5.7 Hyperparameter Tuning

When making a model, we need to find optimal hyperparameters for the model. 

Hyperparameters aren't optimized by the learning algorithm itself. You have to experimentally determine the best value yourself. If you have enough data in your validation set, you can perform **grid search**.

Grid search is the most simple **hyperparameter tuning** technique. Imagine you are training an SVM and there are two hyperparameters: the penalty parameter $C$ and the kernel, either "linear" or "rbf".

You can start with a logarithmic scale for C to start with, then narrow your search down: [0.001, 0.01, 0.1, 1, 10, 100, 1000]. In this case, there are $7 \times 2$ combinations of hyperparameters to test, I hope you can see why. 

You have to train 14 models, one for each combination of hyperparameters, then assess the performance of each. When one model outperforms the other, you can narrow down your search, and repeat the same process.

This can be time consuming, for very large models and datasets. **Random search** and **Bayesian hyperparameter optimization**. 

In random search, you no longer has a discrete set of values to explore for each hyperparameter. Instead, you provide a statistical distribution for each hyperparameter from which values are randomly sampled and set the total number of combinations you want to try.

The bayesian technique differs from the random or grid search in that they use past evaluation to choose the next values to test. It limits the number of optimizations of the objective function by choosing new values from values that have done well in the past.

**Gradient-based techniques** and **evolutionary optimization techniques** are also popular, but too advanced to explain here.

### Cross-Validation

If you have a large validation set, you can use **cross-validation**. When you have limited training examples, it could be a detriment to both the training and validation sets. You want to use more data to train the model - you can split your data into training and test, and use the full training set to simulate a validation set. 

The algorithm works as follows. First you fix the values of the hyperparameters you want to evaluate. Then you split your training set into severl subsets of the same size. Each subset is called a **fold**. With five-fold cross validation, you randoml split your training data into five folds: ${F_1, F_2, F_3, F_4, F_5}$. Then you train five models. To train the first model, $f_1$, you use all examples from folds $F_2, F_3, F_4, F_5$ and set the examples from $F_1$ to be the validation set. to train $f_1$, you use all examples from folds $F_1, F_3, F_4, F_5$ and set the examples from $F_2$ to be the validation set. 

Continue to build these models, and compute the value of the metric of interest on each validation set, and then average that to get the final value.

# Chapter 6: Neural Networks and Deep Learning

A neural network is a logistic regression model, a generalization for multiclass classification is called the softmax regressor model, and is standard in most neural networks.

## 6.1 Neural Networks

A **neural network** (NN) just like any other model, is a mathematical function:
$$
y = f_{NN}(x)
$$

The function$f_{NN}$ has a particular form: it's a **nested function**. Neural networks have **layers**. A network that has three layers can look like this:
$$
y = f_{NN} = f_3(f_2(f_1(x)))
$$

In the equation above, $f_1$ and $f_2$ are vector functions in the form:
$$
f_l(x) = g_l(W_lx+b_l)
$$
where $l$ is the layer index. The function $g_l$ is called the **activation function**. It's a fixed function chosen by the user before the learning is started. The parameters $W_l$ (matrix) and $b_l$ (vector) for each layer is learned using gradient descent optimization depending on the task, on a particular loss function. If you replace ${g_l}$ with the sigmoid function, it's is identical to logistic regression.

$g_l$ is a vector function, Each row of $W_l$ is a vector the same dimensionality as the vector $x$. The output of $f_1(x)$ is a vector, where $g_l$ is some scalar function. To make this more clear, let's consider one architecture of neural networks, called **multilayer perceptron** and is usually called a **vanilla neural network**.

### Multilayer Perceptron Example

Let's take a look at a particular configuration of neural networks called **feed-forward neural networks**, and more specifically the **multilayer perceptron**. Let's consider an MLP with three layers. The networks takes a two-dimensional feature vector and outputs a number. This FFNN can be for a regression or classification depending on the activation functions used in the third output layer.

The neural network is represented as a connected combination of **units**, organized into **layers**. Each unit is represented graphically by a circle or a rectangle. In each unit, all the inputs of the unit are joined together to form an input vector. Then the unit applies a linear transformation to the input vector. The the function applies an activationg function $g$ to the result of the linear transformation and obtains the output value, a real number. 

In multilayer perceptron, all outputs of one layer are connected to each input of the succeeding layer. This architecture is called **fully-connected**. A neural network can contain **fully connected layers**. 

### Feed-Forward Neural Network Architecture

If we want to solve a regression or classification problem, the last layer of a neural network will contain one unit. If the activation function of the last unit is linear, the nueral network is a regression model. If the last activation function is a logistic function, the neural network is a classification model.

Any activation function can be chosen, but being differentiable is nice. The main purpose of having non-linear components in the function $f_{NN}$ is to allow the neural network to approximate non linear functions. Without then, neural networks would bel inear, no matter how many layers it has. 

Other popular activation functions include the logistic function, and the **TanH** (hyperbolic tangent) and **ReLU** (rectified linear unit).
$$
tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}} \\
\text{}\\
relu(z) = 
\begin{cases} 
0 & \text{if } z < 0 \\
z & \text{otherwise} 
\end{cases}
$$

In matrix $W_l$, each row $u$ corresponds to a vector of parameters $W_{l, u}$. The dimensionality of the vectr $w_{l, u}$ equals the number of units in the later $l-1$. The operation $W_lz$ results in a vector $a_l = [w_{l,1}z, w_{l,2}z, w_{l,3}z, ..., w_{l,size}z]$. The sum $a_l + b_l$ return a size($l$) dimensional vector $c_l$. Finally, the function $g_l(c_l)$ produces the vector $y_l$ as the output. Don't worry if this is hard to follow, it's supposed to be.

## 6.2 Deep Learning

**Deep learning** refers to training neural networks with more than two non-output layers. The two largest issues in deep learning were **exploding gradients** and **vanishing gradients** as gradient descent was used. 

While the problem of exploding gradient can be solved through **gradient clipping** and regularization, the vanishing gradient issue remained.

What is vanishign gradient? To update the values of the parameters in neural networks, **backpropogation** is used. It's a fast algorithm for computing gradients on neural networks using the chain rule. During gradient descent, the neural network's parameters recieve an update proportional to the partial derivative of the cost function with repsect to the current parameter in each iteration. Sometimes, the gradient will be vanishingly small, effectively preventing parameters from being updated at all. This may completely stop the neural network from training.

Traditional activation functions, such as the hyperbolic tangent have gradients in the range $(0,1)$ and backpropagation computes gradients by the chain rule. That has the effect of multiplying $n$ of these small numbers to compute gradients of the earlier layers. The gradients decrease exponentially with $n$ in an $n$-layer network. The earlier layers would train very slowly, if at all.

However, the modern implementations of neural network learning algorithms allow you to train very deep neural networks. This is due to many improvements, including ReLU, LSTM, as well as techniques including **skip connections** used in **residual neural networks**.

The term deep learning today refers to training neural networks using the modern algorithmic and mathematical toolkit, regardless of how deep a network really is. The layers that are neither input nor output are often called **hidden layers**.

### Convolutional Neural Network

The number of parameters in an MLP can grow very fast as you make the network bigger. Optimizing a large model is very computationally intensive. 

When dealing with images, the input is very high dinensional. Classifying images with an MLP is virtually impossible. A **convolutional neural network** is a special kind of FFNN that significantly reduces the number of parameters in a deep neural network with many units without losing too much quality of the model. 

Let's consider image processing throughout this section. In images, pizels that are close to one another usually represent the same kind of information: sky, water, leaves, fur, etc. The exceptions: edges. The parts of an image where two different object "touch" one another. 

If a neural network can recognize regions of the same information and the edges, this would allow the network to predict the object represented in the image.

We can split the image into square patches, using a moving window approach. We can then use this information to train multiple smaller regrssion models at once, each smaller regression model will learn to detect the sky, the grass, or the building, so on. In CNNs, a small regression model only has one layer. To detect patterns, the small regression model has to learn the parameters of a matrix **F** (filer) of size $p\times p $, where $p$ is the size of a patch.

Let's assume the input image is black and white, with 1 representing blakc pixels, and 0 representing white pixels. Assuming that the patches are 3 by 3 pixels, a patch $P$ could look like this:
$$
P = \begin{bmatrix}
0 & 1 & 0\\
1 & 1 & 1\\
0 & 1 & 0
\end{bmatrix}
$$

the patch above has a cross, and a smaller regression model would detect that. If we calculate the **convolution** of matrices $P$ and $F$, the value we obtains is higher the more similar $F$ is to $P$. Assume that $F$ looks like this:

$$
F = \begin{bmatrix}
0 & 2 & 3\\
2 & 4 & 1\\
0 & 3 & 0
\end{bmatrix}
$$

I would recommend looking up the convolution calculation, but it is the sum of the element-wise products of the two matrices. So the convolution of $P$ and $F$ would be 12. If we were looking at a different path, say one with the shape of the letter L:

$$
P = \begin{bmatrix}
1 & 0 & 0\\
1 & 0 & 0\\
1 & 1 & 1
\end{bmatrix}
$$

then the convolution with $F$ would be 5, a significantly lower result. There is also a bias parameter $b$ associate with each filter $F$ which is added to the result of a convolution before the activation function.

One layer of a CNN consists of multiple convolution filters - just like one layer in a vanilla FFNN consists of multiple units. The filter matrix, also commonly called the **kernel** and bias values are trainable paramters that are optimized using gradient descent with backpropagation.

A nonlinearity, as mentioned before, is applied to the sum of the convolution and bias terms. Since we can have $size_l$ filters in each layer $l$, the output of the convolutional layer $l$ would consists of $size_l$ matrices. This sounds confusing, but if you read through it again, it'll make sense.

If the CNN has one convolution layer following another convolution layer, then the subsequent layer $l+1$ treats the output of the preceding layer $l$ as a collection of $size_l$ images. This collection of images (matrices) is called a **volume**. The size of that collection is called the volume's **depth**. Every filter of layer $l+1$ convolves the *entire* volume. The patch of a volume is just the sum of the convolutions of the corresponding patches of the individual matrices that make up the volume.

CNNs actually usually get volumes as an input, as an image is decomposed into three channelsL: R, G, and B, each channel being a monochrome image. Images can start as volumes of depth 3.

Two important properies of convolution are **stride** and **padding**. Stride is the step size of the moving window. If the stride is 1, the filter slides from right and to the bottom by one cell at a time. The output matrix is smaller when the stride is larger.

Padding allows getting a larger output matrix, it's the width of the square of additional cells that surround the image before it's convolved with the filter. The cells that are a part of padding usually have a value of zero. Padding is helpful with larger filters because it allows them to scan the boundaries (corners and edges) of the image.

**Pooling** is essential to CNNs. Pooling works in a way similar to a convolution, as a filter applied using a moving window approach. But instead of applying a trainable filter to an input matrix or a volume, pooling layers apply a fixed operator, usually the max or the average. Pooling also has hyperparameters: the size of the filter and the stride. 

A pooling layer usually follows a convolution layer, and it gets the output of convolutions as an input. When pooling is applied to a volume, each matrix in the volume is processed independently of others. Therefore, the output of a pooling layer applied to a volume is a volume the same depth as the input.

Pooling usually contributes to an increased accuracy of the model. It improves the training speed of the model by reducing the number of parameters in the neural network.

### Recurrent Neural Network

**Recurrent Neural Networks** (RNNs) are used to label, classify, or generate sequences. A sequence is a matrix each row of which is a feture vector and the order of rows matter. Labelling a sequence is to predict a class for each feature vector in a sequnce. To classify a sequence is to predict a class for the entire sequence. To generate a sequence is to output another sequence somehow related to the input sequence.

RNNs are used in text processing because sentences are naturally sequences of words and charaters. 

A recurrent neural network is nor feed-forward: it contains loops. The idea is that each unit $u$ of a recurrent layer $l$ has a real-vlued **state** $h_{l, u}$. The state can be seen as the memory of the unit. In RNNs, each unit $u$ recieves two inputs: a vector of states from the previous layer, and the vector of states from the same layer from the *previous time step*. 

Let's break this down. Consider the first two layers of a trivial RNN. The first layer recieves a feature vector as input. The second layer recieves the output of the first layer as the input. Pretty simple.

Each training example is a matrix in which each row is a feature vector. Let's illustrate this matrix as a sequence of vectors $X = [x^1, x^2, ..., x^{t-1}, x^{t}, x^{t+1}, ... x^{length x}]$ where $length x$ is the length of the input sequence. If our input is a text sentence, then the feature vector $x^t$ for each $t = 1, ..., length x$ represents a word at position $t$.

The feature vectors from an input example are "read" by the neural network sequentially in the order of the timesteps. The index $t$ denotes a timestep. To update the state $h^t_{l, u}$, at each timestep $t$ in each unit $u$ of layer $l$, we first calculate a linear combination of the input feature vector with the state vector $h^{t-1}_{l, u}$ of the same layer from the previous timestep $t-1$. The linear combination of the two vectors is calculated using the two parameter vectors $w_{l, u}, b_{l, u}$ and a parameter $b_{l, u}$. The value of $h^t_{l, u}$ is then obtained by applying an activation function $g_1$ to the result. The output vector $y^t_l$ is typically a vector calculated for the whole layer $l$ at once. TO obtain the output vector $y$, we use the activation function $g_2$ that takes a vector as input and returns a different vector of the same dimensionality. The function $g_2$ is applied to a linear combination of the state vectors $h^t_{l, u}$ calculated using a parameter matrix $V_l$ and a parameter $c_{l, u}$. In classification a typical choice for $g_2$ is the **softmax function**:

$$
\sigma{z} = [\sigma^{(1)}, ..., \sigma^{(D)}] \text{ where } \sigma^{(j)} = \frac{\exp(z^{(j)})}{\sum^D_{k=1}\exp(z^{(k)})}
$$

The softmax function is a generalization of the sigmoid function to multidimensional outputs. 

The dimensionality of $V_l$ is chosen by the user such that the multiplication of $V_l$ with $h^t_l$ results in a vector of the same dimensionality as the vector $c_l$.

The values of $w_{l, u}, u_{l, u}, b_{l, u}, V_{l, u}, and c_{l, u}$ are computing from training data using gradient descent with backpropagation. To train RNN models however, a special version of backpropagation, **backpropagation through time** is used.

Both $tanh$ and $softmax$ summer from teh vanishing gradient problem. Even if our RNN only has one or two reccurent layers. Backpropagation has to unfold over time. The longer the input sequence, the deeper the unfolded network.

RNNs are also not great at capturing long term dependencies. As the lenght of an input sequence ggrows, the feature vectors at the beginning of the sequence begin to be "forgotten", because the state of each unit, which serves as memory, becomes significantly more affected by the feature vectors read more recently. 

The most effective reccurent neural network models used in practie are **gated RNNs**, this includes **long short-term memory** (LSTM) networks and networks based on the **gated reccurent unit** (GRU).

The beauty of using gated RNNs is that such networks can store information in their units for future use, kinda like bits in a computer's memory. The difference with the real memory is that reading, writing, and the erasure of information is controlled by activation functions that take values in the range $(0,1)$. 

The trained neural network can "read" the input sequence of feature vectors and decide at some timestep to remember som information. That information can then be used to process feature vectors towards the end of the sequence. 

Units make decisions about what information to store, and when to allow read, write and erasures. Those decisions are learned from data and implemeneted through the concept of *gates*. A **minimal gated GRU** is a popular example, and contains a memory cell and a forget gate.

Here's the math of a GRU unit on the first layer of an RNN. A minima gated GRU unit $u$ in layer $l$ takes two inputs: the vector of the memory cell values from all units in the same layer from the previous timestep, $h^{t-1}_l$, and a feature vector $x^t$. It then does the following calculations in the order listed:
$$
\tilde{h}^t_{l, u} = g_1(w_{l, u}x^t + u_{l, u}h^{t-1}_l+b_{l, u}) \\
\text{} \\
\Gamma^t_{l, u} = g_2(m_{l, u}x^t + o_{l, u}h^{t-1}_l+a_{l, u}) \\
\text{} \\
\tilde{h}^t_{l, u} = \Gamma^t_{l, u}\tilde{h}^t_{l, u} + (1-\Gamma^t_{l, u})h^{t-1}_l \\
\text{} \\
h^{t}_l = [h^{t}_{l, 1}, ..., h^{t}_{l, size}]\\
\text{} \\
y^t_l = g_3(V_lh_l^t+c_{l, u})
$$

where $g_1$ is the $tanh$ function, $g_2$ is the gate function, and is implemented as the sigmoid function taking values in the range $(0,1)$. If the gate function is close to zero, then the memory cell keep sits value from the previous timestep, otherwise the value of the memory vell is overwritten by a new value $\tilde{h}^t_{l, u}$. $g_3$ is usually a $softmax$ function.

A gated unit takes an input and stores it for some time, like the identity function ($f(x) = x$). Because the derivative of the identity function is constant, when a network with gated units is trained with backpropagation through time, the gradient does not vanish.

More popular RNNs include **bi-directional RNNs**, RNNs with **attention**, and **sequence-to-sequence RNNs**. All RNNs can be generalized as a **recursive neural network**.

# Chapter 7: Problems and Solutions

## 7.1 Kernel Regression

What if the data we have doesn't fit a straight line? Polynomial regression could help. Assume we have one-dimensional data $\{(x_i, y_i)\}^N_{i=1}$. We could try to fit a quadratic line $y = w_1x_i^2 + w_2x_i+b$ to our data. We could use gradient descent with backpropagation on a cost function. But if our input is a D-dimensional feature vector, finding just a polynomial would be virtually impossible.

**Kernel Regression** is a non-parametric method. There are no parameters to learn. The model is based on the data itself, kind of like a kNN. The model would look like this:
$$
f(x) = \frac{1}{N}\sum^N_{i=1}w_iy_i, \text{ where } w_i = \frac{Nk(\frac{x_i-x}{b})}{\sum^N_{l=1}k(\frac{x_l-x}{b})}
$$

The function $k(z)$ is called the **kernel**. The kernel plays the role of a similarity function: the values of coefficients $w_i$ are higher when $x$ is similar to $x_i$ and lower when they are dissimilar. Kernels can have different forms, the most frequently used one is the Gaussian kernel:
$$
k(z) = \frac{1}{\sqrt{2\pi}}\exp(\frac{-z^2}{2})
$$

The value $b$ is a hyperparameter that you can tune. 

## 7.2 Multiclass Classification

Some classification algorithms can be defined with more than two classes. In **multiclass classification**, the label can be one of $C$ classes. Many machine learning algorithms are binary, but many can be extended to handle multiclass problems. 

Logistic regression can be naturally extended to multiclass learning problems by replacing the sigmoid function with a **softmax function**.

But when you a multiclass problem and a binary classification learning algorithm, a strategy called **one versus rest** is often used. The idea is to transform a multiclass problem into $C$ binary classification problems and to build $C$ binary classifiers. For example, if we have three classes, we can create copies of the original dataset and modify them. The first copy will have all labels not equal to 1 set to 0. The second copy will have all labels not equal to 2 set ot zero. The third copy will have all the labels not equal to 3 set to zdro. Now we have three binary classification problems. 

Once we have the three models, to classify the new inpout feature vector $x$ we apply the three models to the input, and we get three predictions, and we pick the most certain one. 

## 7.3 One-Class Classification

Sometimes the dataset has examples of one class and we want to train a model what would distinguish examples of this class from everything else. 

**One-class classification**, also known as **unary classification** or **class modeling** tries to identify objects of a specific class among all objects, by learning from a training set contianing only objects of that class. This is harder than regular classification. One-class classification learning algorithms are used for outlier detection, anomaly detection, and nolvelty detection.

There are several one-class learning algorithms: **one-class Gaussian** , **one-class k-means**, **one-class kNN**, and **one-class SVM**.

The idea behind the one-class Gaussian is that we model our data as the Gaussian distribution, a **multivariate normal distribution**. The probability density function for MND is:
$$
f_{\mu, \Sigma}(x) = \frac{\exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))}{\sqrt{(2\pi)^D}|\Sigma|}
$$

where the function $f$ rturns the probability density corresponding to the input feature vector $x$. Values $\mu$, $\Sigma$ are the learnable paramters, a vector and a matrix. The **maximum likelihood** criterion is optimized to find the optimal values for these two parameters. 

The vector $\mu$ determines where the curve of our Gaussian distribution is centered, while the numbers in $\Sigma$ determine the shape of the curve. Once the model is parameterized by $\mu$ and $\Sigma$, we predict the likelihood of every input $x$ using $f$. Only if the likelihood is above a certain threshold, we predict that the example belongs to that class, else it's an outlier. 

When the data has a more complex shape, we need a more complex algorithm with multiple Gaussians, There are going to be more parameters to learn from the data: one $\mu$ and one $\Sigma$ for each Gaussian and the parameters that combines multiple Gaussians to form one probability density function. 

One-class k-means and one-class kNN are based on a similar principle: build some model of the data and then define a threshold to decide whether our new deature vector looks similar to other examples. I will introduce the **k-means** clustering algorithm. When a new example $x$ is observed, the distance is calculated as the minimum distance between $x$ and the center of each cluster. If the distance is less than a particular threshold, then $x$ belongs to that class. This can be generalized to any number of classes. 

A One-class SVM, depending on the specific model, tries to: 1) separate all training examples from the origin, and maximize the distance from the hyperplane to the origin, or 2) to obtain a spherical boundary around the data by minimizing the volume of this hypersphere. 

## 7.4 Multi-Label Classification

When more than one label is appropriate to describe and example from a dataset, it is called **multi-label classifiction**. A picture of a mountain could have the labels: "conifer", "mountain", and "trail" associated with it all at once. 

If the number of possible values for labels is high, but they are all of the same nature, we can transform each labeled example into several labeled examples, one per label. These new examples all have the same feature vector and only one label. That becomes a multiclass classification problem, and we can solve it with using the one versus rest strategy. Now, we have a new hyperparameter: threshold. If the prediction score for some label is above the threshold, this label is predicted for the input feature vector. 

Algorithms that cna naturally make multiclass predictions (decision trees, logistic regression, and neural networks) can be applied to multi-label classification problems. Because they all output a score for each class, we can define a threshold and then assign multiple labels to one feature vector if the threshold is above from value. 

Neural networks can naturally train multi-label classification models by using the **binary cross-entropy** cost function. The output layer of the neural network has one unit per label. Each unit of the output layer has the sigmoid activation function. Each label $l$ is binary. The binary cross-entropy of predicting the probability $\hat{y}_{i, l}$ that example $x$ has label $l$ defined as:

$$
-(y_{i, l}\ln(\hat{y}_i, l) + (1-y_{i, l})\ln(1-\hat{y}_{i, l}))
$$


The minimization criterion is simply the average of all binary cross-entropy terms across all training examples and their labels. 

In cases where the number of positive values each label can take is small, one can convert multilabel into a multiclass problem using a different approach. Example: we want to label images and labels can be of two types. The first type of label can have two values: {photo, painting}, the label of the second type can have three possible values: {portrait, paysage, other}. We can create new fake classes for each combination of the two original classes, ending up with $2 \times 3 = $ individual classes total. 

This is fine, but can't be done when there are too many possible combinations of classes. This approach keeps your labels correlated, where previous approaches predict each label independently of one another. 

## 7.5 Ensemble Learning

Right now, we've learned two extremes of the spectrum, the algorithms introduced in chapter 3, and very deep neural networks. An approach to meet in the middle of the complexity number line is **ensemble learning**.

Ensemble learning focuses on making many weak models, not on one very precise model, and them combines the predictions given by those *weak* models to obtain a high-accuracy **meta-model**.

Low-accuracy models are usually learned by **weak learners**, learning algorithms that cannot learn complex models, and therefore are fast at training and at the prediction. The most frequently used weak learner is a decision tree. The obtained trees are shallow and not very accurate. But the idea is, if the trees are not identical and each tree is slightly better than random guessing, then can obtain a high accuracy by combining a large number of these trees.

### Boosting and Bagging

Boosting consists of using the original training data and iteratively creating multiple models by using a weak learner. Each new model would be different from the precious one because as a new model is made, it tries to fix the errors the prvious model made. The final **ensemble model** is a certain combination of those multiple weak models built iteratively. 

Bagging consistss of creating multiple "copies" of the training data, where each copy is slightly different, and then applying the weak learner to each copy to obtain multiple weak models and then combining them. A widely used machine learning algorithm based on the idea of bagging is **random forest**.

### Random Forest

The "vanilla" bagging algorithm works as follows. Given a training set, we create $B$ random samples $S_B$ of the training set and build a decision tree model $f_b$ using each sample $S_B$ as the training set. To sample $S_B$ from $B$, we do the **sampling with replacement**. This means that we start with an empty set, and then pick at random an example from the training set, and put its exact copy into the subset, keeping the original examples in the original training set. We keep picking examples at random until $|S_B| = N$

After training, we will have $B$ decision trees. The prediction for a new examples $x$ is obtained as the average of $B$ predictions
$$
y = \hat{f}(x)=\frac{1}{B}\sum^B_{b=1}f_b(x)
$$

in the case of regression, or by taking the majority vote in the case of classification.

Random forest is different from the vanilla bagging in one small way. It uses a modified tree learning algorithm that inspects, at each split in the learning process, a random subset of the features. It does this to avoid the correlation of the trees: if one or few featuers are very strong predictors for the target, these features will be selected to split examples in many trees. This would result in many correlated trees in our "forest". Correlated predictors will not help in improving the accuracy of the prediction. The main reason behind a better model is that the good models will likely agree on the same prediction, while bad models will disagree on different ones. Correlation will make bad models agree, which would skew the vote / average. 

The most important hyperparameter to tune is the number of trees, $B$, and the size of the random subset of the features to consider each split. 

Why is random forest so effective? The reason is that by using multiple samples of the original dataset, we reduce the **variance** of the final model. Low variance means low **overfitting**. The model doesn't learn the small variations in the dataset because our dataset is just a small sample of the population of possible examples. By creating multiple random samples with replacement, we reduce the effect of artifacts that contribute to overfitting.

### Gradient Boosting

Another effective ensemble learning algorithm, is **gradient boosting**. Let's look at gradient boosting for regression. Let's start with a constant model, that just predicts the average:
$$
f=f_0(x) = \frac{1}{N}\sum^N_{i=1}y_i
$$

We can modify the labels of each example in our training set:
$$
\hat{y}_i = y_i - f(x_i)
$$

where $\hat{y}_i$ is called the **residual**, and it's the new label for example $x_i$. 

Now with the new training set that contains the residuals, we build a new decision tree model, $f_1$. The boosting model is now defined as $f = f_0 + \alpha f_1$, where $\alpha$ is the learning rate. 

Then we recompute the residuals and replace the labels in the training data, and we make yet another decision tree model $f_2$, redefine the boosting model as $f = f_0 + \alpha f_1 + \alpha f_2$ and the process continues until the predetermined maximum of $M$ trees are conbined. 

This may seem like a pointless process, but it's not. Computing the residuals tells us how well the target of each training examples is predicted by the current model $f$, kind of like the error. Then we train another tree to fix the errors of the current model, and add this new tree to the exisiting model with some weight $\alpha$. Each additional tree added partially fixes the errors made by the previous trees until the maximum number $M$ of trees are combined. 

Okay... but this seems like this has nothing to do with gradients. In gradient boosting, we don't calculate any gradients. To see how gradient boosting and descent are similar, let's return to linear regression. We calculate the gradients in linear regression to move the values of our parameters in a direction so that the cost function reaches its minimum. The gradient shows the direction, but and the $\alpha$ hyperparameter dictates how large of a step to take in that direction. The same happens in gradient boosting. But instead of getting the gradient directly, we use its proxy in the form of residuals: it shows how the model needs to be changed to reduce the residuals.

The three main hypereparameters in gradient boosting to tune is the number of trees, the learning rate, and the depths of the trees. 

Training on residuals optimizes the overall model $f$ for the mean squared error criterion. This is the difference with bagging: boosting reduces the bias instead of the variance - boosting is prone to overfitting. But this can be avoided by careful tuning. 

Gradient boosting for classification is similar, but the steps are different. Consider the binary case. Assume we have $M$ regression decision trees. Similar to logistic regression, the prediction of the ensemble of decision trees is modeled using the sigmoid function:
$$
\text{Pr}(y=1|x, f) = \frac{1}{1+e^{-f(x)}}
$$

where $f(x) = \sum^M_{m=1}f_m(x)$ where $f_m$ is a decision tree.

We apply a maximum likelihood principle by trying to find and $f$ that maximizes $L_f = \sum^N_{i=1}ln[\text{Pr}(y_i=1|x, f)]$. Again, to avoid **numerical overflow**, we maximize the sum of the log-likelihoods rather than the product of the likelihoods. 

The algorithm starts with the initial constant model $f = f_0 = \frac{p}{1-p}$, where $p = \frac{1}{N}\sum^N_{i=1}y_i$. Then at each iteration $m$, a new tree $f_m$ is added to the model. To find the best $f_m$, the partial derivative of $g$, of the current model is calculated for each example:
$$
g_i = \frac{dL_f}{df}
$$

where $f$ is the ensemble classifier model built at the previous iteration $m-1$. To calculate $g_i$, we need to find the deriatives of $ln[\text{Pr}(y_i=1|x, f)]$ with respect to $f$ for all $i$. 

We then transform out training set by replacing the original label $y_i$ with the corresponding partial derivative $g_i$, and build a new tree $f_m$ using the transformed training set. Then we find the optimal update step $\rho_m$:
$$
\rho_m = \argmax_{\rho}L_{f+\rho f_m}
$$

At the end of the iteration $m$, we update the ensemble model $f$ by added the new tree $f_m$:
$$
f = f + \alpha \rho_m f_m
$$

We iterate until $m = M$, and then we return the ensemble model. 

Gradient boosting is popular because it can get very accurate, and it's able to handle very large datasets with millions of examples and features. But because of it's sequential nature, can be slower in training.

## 7.6 Learning to Label Sequences

Sequence data is the msot comomonly observed data form. We talk using sequences, time is sequenced, tasks are sequenced, genes, music, and videos are all sequences. 

**Sequence labeling** is the problem of automatically assigning a label to each element of a sequence. A labeled sequential training example in sequence labeling is a pair of lists $(X, Y)$, where $X$ is a list of feature vetors, and $Y$ is a list of the same length of labels. $X$ could represent works in a sentence, and $Y$ could be the corresponding parts of the speech. In an example $i$, $X_i = [x^1_i, x^2_i, ..., x^{size_i}_i]$ where $size_i$ is the length of the sequence, $Y_i = [y^1_i, y^2_i,..., y^{size_i}_i]$.

RNNs can handle sequences, at time step $t$, it reads an input feature vector $x^{(t)}_i$, and th elast recurrent layer outputs a label $y^{(t)}_{last}$.

However RNNs are not the only possible model for sequence labeling. A model called **Conditional Random Fields** (CRF) is a very effective alternative that often performs well in practice for the feature vectors that have many informative features. Imagine we have a task called **named entity extraction** where we want to build a model that labels each work in a sentence, e.g. "I go to San Fransisco", with one of the following classes: {$location$, $name$, $company_name$, $other$}. If the feature vectors contain binary features such as "Does the word start with a capital letter" or "Does the word appear in a list of locations", they would be very helpful. 

But building these features are way too labor-intensive. CRF is an interesting model and can be seen as a genrealization of logistic regression in sequence data. But usually RNNs are able to outperform them. They are also slower in training making them hard to use on huge datasets. 

## 7.7 Sequence-to-Sequence Learning

**Sequence-to-Sequence Learning** is a generalization to the sequence labeling proble. In seq2seq, $X_i$, and $Y_i$ can have different lengths. seq2seq is commonly used in machine learning translation, conversatinal interfaces, text summarization, spelling correction, and others. 

Not all seq2seq learning problems are solved by neural networks. The network architectures used in seq2seq all have two parts: an **encoder** and **decoder**.

In seq2seq neural network learning, the encoder is a neural network that accepts a sequence as an input. It can be an RNN or a CNN or some other architecture. The role of the encoder is to read the input and generate some sort of state that can be seen as a numerical representation of the *meaning* of the input sequence. The meaning of some input sequence is usually a vector or matrix that contains real numbers, this vector or matrix is called the **embedding** of the input. 

The decoder is another neural network that takes an embedding as input and is capable of generating a sequence of outputs. The embedding it takes is from the encoder. To produce a sequence of outputs, the decoder takes a *start of sequence* input feature, $x^{(0)}$, usually a vector of zeros, and produces the first output $y^{(1)}$, updates its state by combining the embedding and the input $x^{(0)}$, and then uses the output $y^{(1)}$ as its next input $x^{(1)}$. 

Both the encoder and the decoder are trained simultaneously using the training data. The errors at the decoder output is backpropagated through the entire model. 

More accurate predictions can be obtained with architectures that use **attention**. The attention mechanism is implemented by an additional set of parameters that combine some information from the encoder and the current state of the decoder to generate the label. This allows for better retention of long term dependencies than provided by gated units and a bidirectional RNN.

This is very new, and new models are coming out everyday, so keep an eye on it. 

## 7.8 Active Learning

**Active Learning** is an interesting supervised learning paradigm. It's used when obtaining labels are expensive. This is usually the case because medical records and financial data is hard to get, or when the opinion of an expert may be required to annotate data. THe idea is to start learning with relatively few labeled examples, and a large number of unlabeled ones. 

There are multiple strategies to active learning. here, we discuss only the following two:

1) data density and uncertainty based
2) support vector-based

The first strategy applies the current model $f$, trained using the existing labeled examples, to each of the tramaining unlabelled examples. For each unlabeled examples $x$, the importance score is calculated: $density(x) \cdot uncertainty_f(x)$. Density reflects how many examples surround $x$ in its close neighborhood, while the uncertainty reflexts how uncertain the prediction of the model $f$ is for $x$. In binary classification with sigmoid, the closer the prediction is to 0.5, the more uncertain the model is for that prediction. In SVM, the closer the example is to the decision boundary, the more uncertain the prediction. 

In multiclass classification, **entropy** can be used as a measure of uncertainty:
$$
H_f(x) = -\sum^C_{c=1}\Pr(y^{(c)};f(x))\ln[\Pr(y^{(c)}; f(x))]
$$

where $\Pr(y^{(c)};f(x))$ is the probability score the model $f$ assigns to class $y^{(c)}$ when classifying $x$. If for each $y^{(c)}$, $f(y^{(c)}) = \frac{1}{C}$, basically just guessing, then the model is most uncertain and the entropy is at its maximum of 1. If for some $y^{(c)}$, $f(y^{(c)}) = 1$, then the model is very certain about the class $y^{(c)}$ and the entropy is at its minimum of 0.

Density for the example $x$ can be obtained by taking the average of the distance from $x$ to each of it's $k$ nearest neighbors. 

Once we have the importance score, of each example, we pick the one with the highest importance score and either buy it, or ask an expert to label is. Then we rebuild the model with the labeled example, and continue the process until some criterion is satisfied. This criterion can be the maximum number of requests, or it can depend on how well your model performs. 

The support vector-based active learning strategy consists in build an SVM model using the labeled data. We then ask experts to label the unlabeled example(s) that lie closest to the hyperplane. The idea is that if the example is close to the hyperplane, the most is least certain and would contribute most to the reduction of possible places where the optimal hyperplane could lie. 

## 7.9 Semi-Supervised Learning

In **semi-supervised learning** (SSL), we also have labeled a small fraction of the dataset while most of the reamining examples are unlabaled. We want to use a large number of unlabeled examples to inprove the model performance without asking for additional labeled examples. 

There are many methods for this. One of them, called **self-learning**, uses a learning algorithm to build the initial model using the labeled examples. Then we use the model to all unlabaled examples and label them using the model. If the confidence score of some prediction is higher than some threshold, then we add this labeled example to out training set, retrain the model, and stop until either the performance is satisfactory, or if the performance has stalled. 

This method can help, but it's not great, and the increase in performance isn't super impressive. A better architecture that can be used is the **ladder network**. To understand ladder networks you have to understand what an **autoencoder** is.

An autoencoder is a feed-forward neural network with an encoder-decoder architecture. It's trained to reconstruct its input. So the training example is a pair, $(x, x)$. We want the output $\hat{x}$ of the model $f(x)$ to be as similar to the input $x$ as posible.

THe autoencoder's network looks like an hour glass with a **bottleneck layer** in the middle that contains the embedding of the D-dimensional input vector; the embedding layer usually has much fewer units than $D$. The goal of the encoder, that's in the autoencoder, is to reconstruct the input feature vector from this embedding. Look up an image for the architecture of an autoencoder to understand this.

A **denoising autoencoder** corrupts the left-hand side $x$ in the training example $(x, x)$ by adding some random perturbation to the features. If our examples are grayscale images with pixels represented at values between 0 and 1, a **Gaussian noise** is added to each feature. 
$$
n^{(j)} ~ \Nu(\mu, \sigma^2)
$$

where ~ means sampled from, and $\Nu(\mu, \sigma^2)$ is the Gaussian distribution, with mean $\mu$ and standard deviation $\sigma$, whose probability distribution function is given by:
$$
f_{[\mu, \sigma]}(z) = \frac{1}{\sigma \sqrt(2\pi)} \exp(-\frac{(z-\mu)^2}{2\sigma^2})
$$

The new corrupted by the value of the feature $x^{(i)}$ is given by $x^{(i)}+n^{(j)}$. 

A **ladder network** is a denoising autoencoder with an upgrade. The encoder and the decoder have the same number of layers. The bottle neck layer is directly used to predict the label using softmax. The network has multiple cost functions. For each layer $l$ of the encoder and the corresponding layer $l$ of the decoder, one cost $C^l_d$ penalizes the difference bewteen the outputs of the two layers. When a labeled example is used in training, another cost function $C_c$ penalizes the error in the prediction of the label. The combined cost function, $C_c + \sigma^L_{l=1}\lambda_lC^l_d$, is optimized by the minibatch stochastic gradient descent with backpropagation. The hyperparameters $\lambda$ for each layer determines the tradeoff between the classification and encoding-decoding cost. 

In the ladder network, not only is the input corrupted with noise, but also the output of each encoder layer. When we apply the trained model to the new input $x$, we do not corrupt the input at all. 

Another technique is called S3VM, based on the SVM. We build one SVM model for each possible labeling of unlabeled examples and then we pick the model with the largest margin. 

## 7.10 One-Shot Learning

**One-shot learning** is another important supervised learning paradigm. In one-shot learning, typically applied in face recognition, we want to build a model that can recognize that two photos of the same person represent the same person. If we give the model two different faces, we want it to recognize that they are two different people. 

We could build a binary classifier that takes two images as input and predicts either true or false. But this would create a neural network that twice as big as a regular neural network. Each image would need its own embedding subnetwork. Training such a network would be challenging not only because of its size but also because the positive examples would be much harder to obtain than negative ones. 

A **siamese neural network** can be used to solve this. An SNN can be implemented as any kind of neural network, CNN, RNN, or MLP. THe network only takes one image as input at a time, so the size of the model is not doubled. To obtain a binary classification, out of a network that takes one picture as an input, we train this model differently.

To train an SNN, we use the **triple loss** function. We use three images of a face: image $A$ (for anchor), image $P$ (for positive), and image $N$ (for negative). $A$ and $P$ are two different pictures of the same person; $N$ is a picture of another person. Each training example is not a triplet ($A_i, P_i, N_i$).

The model's task is now to take a picture of a face as input and output an embedding of this picture. The triplet loss for example $i$ is defined as:

$$
\max(||f(A_i)-f(P_i)||^2 - ||f(A_i) - f(N_i)||^2+\alpha, 0)
$$

where $\alpha$ is a positive hyperparameter. When our neural network outputs similar embedding vectors for $A$ and $P$, ||f(A_i)-f(P_i)|| will be low; and when it outputs different embedding vectors for $A$ and $N$, ||f(A_i) - f(N_i)|| will be high. If the model is good, we want the term $m = ||f(A_i)-f(P_i)||^2 - ||f(A_i) - f(N_i)||^2$ to be negative. By setting $\alpha$ higher, we force the term $m$ to be even smaller, to make sure that the model learned to recognized the two faces with a high margin. If $m$ is not small enough, $\alpha$ will make it positive, and the model will continue to make $m$ even smaller. 

Instead of choosing random images for $N$, a better way to create triplets for training to use the current model to find candidates for $N$ similar to $A$ and $P$. Using random samples as $N$ would slow down the training, because the neural network will learn to find the most obvious differences among the individuals. 

To build an SNN, we first decide on the architecture of our neural network. CNN is a popular choice. Given an example, to calculate the average triplet loss, we apply the model to $A$, then $P$, then $N$, and then we compute the loss. We repeat that for all triplets in the batch then compute the cost; then we use gradietn descent to backpropagate the cost through the network. 

You typically want more than one example of each person for the person identification model to be accurate. It's called one shot because of the most important application: face-recognition. Such a model could be used to unlock your phone. If the model is good, you only need to have one picture of you on your phone and it will recognize you, and it can also recognize that someone else is not you. When we have trained the mode, to decide whether $A$ and $\hat{A}$ are the same person, we check is $||f(A) - f(\hat{A})||^2$ is less than $\tau$, a hyperparameter.

## 7.11 Zero-Shot Learning

This last section will be on **zero-shot learning**. It is a relatively new research area, and there are no algorithms that have any practical use yet. In zero-shot learning, we want to train a model that assigns labels to objects, most commonly - labels to images. 

But we want to predict labels that we didn't have in the training data. How would we do that?

You want to use embeddings not just to represent the input $x$ but also the output $y$. Imagine that we have a model that for any word in English can generate an embedding vector with the following property: if a word $y_i$ has a similar meaning to the work $y_k$, then the embeddin vector for these two words will be similar. These embedding vectors are called **word embeddings**, and can be compared using the consine similarity metric.

Each dimensoin of the embedding represents a specific feature of the meaning of the word. If our word embedding has four dimensions (very little), then these four dimensions could represent: animalness, abstractness, sourness, and yellowness. The word *bee* would have the word embedding [1, 0, 0, 1], the word yellow would have [0, 1, 0, 1]. 

Now in our classification problem, we can replace the label $y_i$ for each example $i$ in our training set with its word emebedding and train a multi-label model that predicts word embeddings. To get the label for a new example $x$ we apply out model $f$ to $x$ and get the embedding $\hat{y}$ and look for all English words that have embeddings similar to $\hat{y}$. 

Why does that work? Take a zebra, clownfish, and tiger. Although they are different colors, two are mammals, they are all striped. If these three features are in word embeddings, the CNN would learn to detect these same features in images. Even if the label "tiger" was not in the training set, when "zebra" and "clownfish" are, the CNN would learn the notion of mammalness, orangeness, and the stripness to predict those objects. Once we present the image of a tiger, it will recognize those features, whether they are positive or negative, and the closest word emebedding from the dictinary to the predicted embedding would be tiger. 

# Chapter 8: Advanced Practice

These techniques aren't more complex, but are applied in very specific contexts. 

## 8.1 Handling Imbalanced Datasets

Sometimes, examples of some classes will be underrepresented in your training data. For examples, bank frauds are less common, so there are less of them to use for training. If you use a SVM with soft margins, you can define a cost for misclassified examples. Because noise is always present in the data, there is a hgh chance that some examples of genuine transactions will end up on the wrong side of the decision boundary. 

The SVM algorithm tries to move the hyperplane to avoid misclassified examples as much as possible. The "fraudulent" examples, which are the minority, has risk of being misclassified to classify more of the majority class properly. This is when you have an **imbalanced dataset**. 

If a cost is introduced in misclassifying the minority examples, the model will try harder to avoid misclassifying the minority, and will lead to the cost of misclassification of some examples in the majority class. 

If a learning algorithm doesn't allow weighting classes, you can try the tecnique of **oversampling**. This is where you increase the importance of examples of some class by making multiple copies of the example in that class. This is risky. 

The opposite approach **undersampling**, is to randomly remove some of the majority class that is randomly sampled. 

You can also create new synthetic examples, by randoml sampling feature values from multiple examples from the minority class, and then combine them to obtain a new example of that class. Two popular algorithms are used: *synthetic minority oversampling technique* (**SMOTE**), and the *adaptive synthetic sampling method* (**ADASYN**).

SMOTE and ADASYN work similarly in many ways. For a given example $x_i$ of the minority class, they pick $k$ nearest neighbors of this example and then create a synthetic example $x_{new}$ as $x_i + \lambda(x_{zi} - x_{i})$ where $x_{zi}$ is one of those neighbors. 

Both SMOTE and ADASYN pick all possible $x_i$ in the dataset. In ADASYN, the number ofsyntehtic examples generated for each $x_i$ is proportional to the number of examples in $S_k$, which are not from the minority class. This causes more synthetic examples to be generated where the examples of the minority class are rare.

## 8.2 Combining Models

Ensemble algorithms usually combines models of the same nature. They boost performance by combining hundreds of weak models. We can sometimes get additional performance gain by combining strong models made with different learning algorithms. 

The three ways are 1) averaging, 2) majority vote, and 3) stacking

**Averaging** works for regression and classification models that return confidence scores. You simply use all your base models on the input eaxample, the **base models** and then you average the predictions. 

**Majority Vote** works for classification models. You apply all your base models and then return the majority class among all predictions. If there is a tie, you can randomly pick one, or have an odd number of models to use. 

**Stacking** consists of building a meta-model that takes the output of base models as input. Let's say you want to combine classifiers $f_1$ and $f_2$, both predicting the same set of classes. To create a training example, $(\hat{x_i}, \hat{y_i})$ for the stacked model, set $\hat{x_i} = [f_1(x), f_2(x)]$ and $\hat{y_i} = y_i$. 

To train the stacked model, it's recommended to use the training set and tune the hyperparameters of the stacked model using cross-validation. 

All three of the methods above, make sure that they are actually improving model performance, if not, don't bother. The reason that combining multiple models *can* bring better performance is that several uncorrelated strong models agree that they are more likely agree on the correct outcome. They must be *uncorrelated*. These base models should be obtained using different features or algorithms - e.g. Random Forest and SVM. Combining different versions of the decision tree learning algorithm, or several SVMs, may not improve performance that much.

## 8.3 Training Neural Networks

In neural network training, the hardest thing is to convert your data into the input the network can read and work with. If you have images, they have to be the same size. 

Text has to be tokenized. For CNN and RNN, each token is converted into a vector using one-hot encoding. Another way to represent tokens is by using **word embeddings**. 

Choosing a specific is a difficult choice. Like seq2seq learning, there are multiple architectures you can use. 

The advantage of a modern architecture over an older one becomes less significant as you preprocess, clean and normalize your data, and create a larger training set. These current models can be very hard to implement on your own and usually require much computational power to train. 

Once you have decided on the architecture, you want to decide on the number of layers, their type, and size. Start simple - one or two layers - and then see how it performs, you wan to have a low bias and perform well on the training set. If not, slowly increase the size of the model until it fits the training data perfectly. Once this is achieved, if the model has high variance and doesn't perform well on validation data, add regularization to your model. If adding regularization causes the model to not fit the training data, increase the size of the network.

## 8.4 Advanced Regularization

In neural networks, beside L1 and L2 regularizatiion, there are other neural network specific regularizers: **dropout**, **early stopping**, and **batch-normalization**. 

Dropout is simple. Each time you run a training example through the network, you randomly exclude some units from the computation. The higher the perecentage of units excluded, the higher the regularization effect. You can add a dropout layer between the two successive layers, or you can specify the dropout parameter for the layer. 

Early stopping is the way to train a neural network by saving the preliminary model after every epoch and assessing the performance of the preliminary model on the validation set. As the number of epochs increase, the cost decreases. The decreased cost means that the model fits the training data well. But at some point, it will start overfitting. You can stop the training once you observe a decreased performance on the validation set if you save the version of a model every epoch. Or you can keep running and training process for a fixed number of epochs and then pick the best model. Models saved after each epoch are called **checkpoints**. 

Batch normalization is a technique that standarizes the outputs of each layer before the units of the subsequent layer receive them as input. Batch normalization results in faster and more stable training. 

Another technique that can be used on any learning algorithm is **data augmentation**. This is used to regularize models that work with images. Once you have your original labeled training set, you can create synthetic examples from an original example by applying different transformations of the original image: zooming, rotating, flipping, darkening, etc. all while keeping the original label. This increases the amount of data you have and usually the performance of the model. 

## 8.5 Handling Multiple Inputs

When working with multimodal data, (e.g. images and text, video and audio), it's hard to adapt **shallow learning** algorithms to work with multimodal data. But it's not impossible. Let's say your input is an image and a piece of tet and the binary output indicates whether or not the text describes the image. You could train a shallow model on the image and another one on the text, and then use a model combination model we discussed above.

If you can't divide your problem into two independent subproblems, you can try to vectorize each input, and then concatenate the two vectors together. 

With neural networks, you have more flexibility. You can built two subnetworks, one for each type of input. A CNN could read the image, and an RNN could read the text. Both subnetworks would output an embedding in the their last layer: CMM has an embedding of the image, and the RNN has an embedding of the text. You can then concatenate the two embeddings then add a classification layer, on top of the concatenated embeddings. 

## 8.6 Handling Multiple Outputs

In some cases you might want to output multiple outputs for one input, like in multi-label classification. Most problems that require multiple outputs can usually be simplified into a multi-label classification. Tasks that have labels of the same nature or fake labels and be created as an enumeration of combinations of the original labels. 

In some cases, the outputs are multimodal, and their combinations cannot be enumerated. Take this example: you are building a model that detects and object on an image and returns its coordinates. The model also has to return a tag describing the object. Your training example will be a feature vector that represents an image. The label will be a vector of coordinates and a one-hot encoded tag. 

To handle a situation like this, you can create one subnetwork that would work as an encoder, reading the image with multiple convolution layers. The encoder's last layer would be the embedding of the image. Then you make two other subnetworks: the first has ReLU as the last layer and uses the mean-squared error $C_1$ to calculate the loss, and the second subnetwork will take the same embedding vector as input and predict the probabilities for each tag. The softmax layer can be used for the last layer of the second subnetwork and then use the average negative log-likelihood cost $C_2$ (**cross-entropy**)

Now, with two loss functions, they need to be weighed. You can another hyperparameter $\gamma$ in the range $(0, 1)$ and define the actualy cost function as $\gamma C_1+(1-\gamma)C_2$. 

## 8.7 Transfer Learning

**Transfer Learning** 