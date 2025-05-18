#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Q1 Part 2: Two component GMM  

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

# Step 1: Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

"""
I chose the sample size to be 2000 bc it helped me create the structure of the two moon shaped clusters.
Noise was 0.1 bc otherwise I would not get a well defined moon shape.
Random state was set to zero to ensure that each time we run the code with the same parameters, we will 
end up getting the same dataset. 
Shuffle was set to true to make sure that the data points are not biased and are randomly distributed. 

"""

# Step 2: Build a two-component GMM
gmm = GaussianMixture(n_components=2, random_state=0)
"""
Bc we're building two components, we set n_components=2. As previously mentioned, random state was set 
to zero to ensure that each time we run the code with the same parameters, we will 
end up getting the same dataset.
"""

# Step 3: Fit the GMM to the data
gmm.fit(data_points)

# Step 4: Plot the data and GMM level sets
plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.5, label='Data Points')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='r', marker='x', s=100, label='Means')
"""
The means in GMM serve as the center of the Gaussian components. This helps us understand where the model 
believes the center of the Gaussian distributions are located. 
"""

# Plot level sets of the two Gaussians. I've chosen these values such that they properly showcase the data
# on the graph. 
x = np.linspace(-2, 3, 100)
y = np.linspace(-1.5, 2, 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = -gmm.score_samples(np.c_[X_grid.ravel(), Y_grid.ravel()])
Z = Z.reshape(X_grid.shape)
plt.contour(X_grid, Y_grid, Z, levels=10, cmap='viridis')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model with Two Components')
plt.legend()
plt.show()

# Q1 Part 3: Gaussian mixture as a clustering algorithm

# Step 1: Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)
# The above part was explained before.

# Step 2: Build a two-component GMM
gmm = GaussianMixture(n_components=2, random_state=0)
# The above part was explained before.

# Step 3: Fit the GMM to the data
gmm.fit(data_points)

# Step 4: Predict cluster assignments for each data point
cluster_labels = gmm.predict(data_points)

# Step 5: Plot the data points, each cluster with a different color
plt.scatter(data_points[:, 0], data_points[:, 1], c=cluster_labels, cmap='viridis', alpha=0.4)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='r', marker='x', s=100, label='Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Clustering')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()

# Step 5 was done to ensure that the data points appeared properly in the plot. Trial and error 
# was done to make the data visually appealing. 


# In[3]:


"""
Q1 part 4: 
Q: If you are thinking that a solution is to increase the number of components, how would you choose such
number? There are many ways to do so, one, for instance is to use the BIC criterion on a test set, you
can look here as a starting point for its use with GMM.The number of components should be something
in between 10 and 15. As usual, after you decided how many number of components you need and
explained the procedure for selecting that number, you should plot the level sets of your GMM and
check whether they are compatible with the sample points.

"""

from sklearn.model_selection import train_test_split, GridSearchCV

# Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

# Split data into training and test sets
X_train, X_test = train_test_split(data_points, test_size=0.2, random_state=0)

# Define the parameter grid
param_grid = {
    'n_components': range(10, 16),
    'covariance_type': ['full', 'tied', 'diag', 'spherical']
}
# Grid Search will explore different numbers of components within this range to find the best value.
# Grid Search will explore the combinations of the above mentioned hyperparameters to find the best GMM. 

# Initialize Gaussian Mixture model
gmm = GaussianMixture(random_state=0)

# Perform grid search
grid_search = GridSearchCV(estimator=gmm, param_grid=param_grid, cv=5)
grid_search.fit(X_train)

# The estimator is definied to be gmm bc it represents the GMM that we want to optimize.
# Param_Grid is a dictionary that contains the hyperparameters that we need to tune. During grid search, 
# it will go over all combinations of these hyperparameters to find the best one. 
# The cv parameter tells us the # of folds in a cross validation. cv=5 means that the data will be split into 
# 5 folds, and the grid search will be performed on each fold.  

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# This part gets the best combination of the hyperparameters found durign the search process and keeps track of it.

# Get the best model
best_gmm = grid_search.best_estimator_
# This part gets the best model found during the grid search and keeps track of it. 

# Visualize the clustering results and level sets
x = np.linspace(-2, 3, 100)
y = np.linspace(-1.5, 2, 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = -best_gmm.score_samples(np.c_[X_grid.ravel(), Y_grid.ravel()])
Z = Z.reshape(X_grid.shape)

plt.scatter(data_points[:, 0], data_points[:, 1], c=best_gmm.predict(data_points), cmap='viridis', alpha=0.5, label='Cluster')
plt.scatter(best_gmm.means_[:, 0], best_gmm.means_[:, 1], c='r', marker='x', s=100, label='Component Means')
plt.contour(X_grid, Y_grid, Z, levels=10, cmap='viridis', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Clustering with Full Covariance and Optimal Components (GridSearchCV)')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()


# In[84]:


# Q2: Part 1:
# Plot the data and, if the data set doesn’t come already with labels, assign to each point of the data
# set either the label y = 1 if the point belongs to the upper moon or y = −1 if the point belongs to the
# lower moon (color the points with two different colors depending on the cluster they belong to).

# Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

# Split the data into upper and lower moon
upper_moon = data_points[data_points[:, 1] > 0]
lower_moon = data_points[data_points[:, 1] < 0]
# If the y coordinates are positive then the data point belong to the upper moon.
# If the y coordinates are negative then the data point belong to the lower moon.

# Assign labels: 1 for upper moon, -1 for lower moon
y_upper = np.ones(upper_moon.shape[0])
y_lower = -np.ones(lower_moon.shape[0])
# For the upper moon group, we'll assign the label of 1 to each point. 
# This means that these points belong to the upper moon.
# For the lower moon group, we'll assign the label of -1 to each point. 
# This means that these points belong to the lower moon.


# Plot the data: This part help easily visualize the data.
plt.figure(figsize=(8, 6))
plt.scatter(upper_moon[:, 0], upper_moon[:, 1], color='blue', label='Upper Moon')
plt.scatter(lower_moon[:, 0], lower_moon[:, 1], color='red', label='Lower Moon')
plt.title('Upper and Lower Moon Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()



# In[85]:


# Q2: Part 2:
#You can use a GMM to perform classification using two components, one for each class. After your
#trained your GMM each point can be assigned to a class by looking at P(X|Y = i) where i can be ±1.
#However, it is clear that with only two components the GMM will perform poorly as a classifier. Plot
#the points coloring them according to P(Y |X = x) as we explained in class.

# Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

# Train a Gaussian Mixture Model (GMM) with two components
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_points)

# Compute the posterior probabilities of each class given the data points
posterior_probs = gmm.predict_proba(data_points)
# The above line calculates the probability that each data point belongs to either the 
# upper moon or the lower moon. 
# For each data point, we're basically figuring out how likely it's to belong to each of
# the two moons. 

# Plot the data, coloring them according to the posterior probabilities of both moons
plt.figure(figsize=(8, 6))
plt.scatter(data_points[:, 0], data_points[:, 1], c=posterior_probs[:, 1], cmap='coolwarm', label='Lower Moon')
plt.scatter(data_points[:, 0], data_points[:, 1], c=posterior_probs[:, 0], cmap='coolwarm', label='Upper Moon', alpha=0.5)
plt.colorbar(label='Posterior Probability')
plt.title('Data Points Colored by Posterior Probability')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()




# In[86]:


# Q2 part 3: 
# How can we use your GMM to perform classification when you use more components than classes?
# There are many ways to do so and you can come up with your own (describe it clearly though). A
# starting point could be the following:
# – Assign each component of your GMM to a class. You can do this in many ways; one starting
# point is to evaluate each component at the points of each class separately and define a metric (for
# instance according to a majority rule: within one standard deviation of the component, are there
# more points with y = 1 or y = −1? in the first case assign the component to the class y = 1 or to
# y = −1 otherwise) to identify to which class that component belongs.
# – Given a new observation x (a new point for which we don’t know the label), you can decide
# to which class it belongs by evaluating each component of your GMM at that point (computing
# P(Y = i|X = x) for each component i) and assigning the point to the component corresponding
# to the largest probability.
# – Assign that point to the class that component belongs to.

from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


# Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

# Train a Gaussian Mixture Model (GMM) with two components
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_points)

# Assign components to classes
class_assignments = []
for i in range(2):  # Iterate over components
    # Evaluate each component at points of each class separately
    prob_class_1 = np.mean(gmm.predict(data_points[data_points[:, 1] > 0]) == i)
    prob_class_minus_1 = np.mean(gmm.predict(data_points[data_points[:, 1] < 0]) == i)
    # Assign the component to the class with higher probability
    if prob_class_1 > prob_class_minus_1:
        class_assignments.append(1)
    else:
        class_assignments.append(-1)
"""
The code iterates over each component of the GMM, determining its association with a 
class according to the majority rule. This involves calculating the probability of each 
component being associated with each class, achieved by analyzing the distribution of data 
points within each class separately. After computing these probabilities, the code determines 
whether each component should be linked to class 1 or class -1. These associations are then 
stored in the class_assignments variable for later use.


"""

# Classifying New Observations
def classify_new_point(gmm, class_assignments, new_point):
    # Compute posterior probabilities for each component
    posterior_probs = gmm.predict_proba(new_point.reshape(1, -1))
    # Assign the new point to the class corresponding to the component with the highest probability
    max_prob_index = np.argmax(posterior_probs)
    assigned_class = class_assignments[max_prob_index]
    return assigned_class

"""
Here the function classify_new_point helps to classify new observations based on the trained GMM and the
class assignments that were previously computed. For every new observation, this part of the code calculates 
the posterior probabilities of each component using predict_proba method of the GMM. Also, this part helps
identify the component with the highest posterior probability for every new observation. At the very end, 
we then retrieve the class that is related with that specific component from class_assignments list.
"""


# Testing
new_point = np.array([-1, 1])  # Example new point
predicted_class = classify_new_point(gmm, class_assignments, new_point)
print("Predicted class:", predicted_class)


# In[87]:


# Q2: Part 4: Plot the points together with the level sets of the GMM components and color the points according to
# what class is the GMM assigning them. What do you observe?

# Generate points representing the decision boundaries of the GMM components
x = np.linspace(-2, 3, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape(X.shape)

# Plot the original data points
plt.scatter(data_points[:, 0], data_points[:, 1], c=gmm.predict(data_points), cmap='viridis', alpha=0.5)

# Plot the decision boundaries
plt.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 7), colors='k', alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('GMM Decision Boundaries and Data Points')

plt.colorbar(label='Assigned Class')

plt.show()


# In[88]:


"""
Q2: Part 5: If needed you can increase the number of components of your GMM by minimizing the classification
error. You can do so by dividing the data set in a training set and a test set and increase the number
of components until the error (you can come up with your own metric or get some inspiration from the 
concept of F1 error or weighted empirical error for classification; in any case, justify your choice) on the
test set is at its minimum. Remeber not to overfit the data.


"""



from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Generate data with two moons
data_points, _ = make_moons(n_samples=2000, noise=0.1, random_state=0, shuffle=True)

# Split the data into features (X) and labels (y)
X, y = data_points, _

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Here the test_size=0.2 indicates that the 20% of the data will be used for testing and the 
# remaining 80% will be used for training. 

# Initialize lists to store silhouette scores and number of components
silhouette_scores = []
# This part will store the silhouette scores calculated for different # of components.

num_components_range = range(2, 11)
# This tells the # of components that will be tested.

# Iterate over different numbers of components
for n_components in num_components_range:
    # Train a GMM with the current number of components
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X_train)
# Each time the loop runs, the GMM is trained with the current # of components.
    
    # Predict labels for the test set
    labels = gmm.predict(X_test)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_test, labels)
    
    # Append silhouette score and number of components to the lists
    silhouette_scores.append(silhouette)
# We add the silhouette score and the # of components here.

# Find the optimal number of components with maximum silhouette score
optimal_components = num_components_range[np.argmax(silhouette_scores)]
# We identify the index of the max silhouette score in the list of the silhouette scores.

# Plot the silhouette scores
plt.plot(num_components_range, silhouette_scores, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Components')
plt.xticks(num_components_range)
plt.grid(True)
plt.show()

print("Optimal Number of Components:", optimal_components)
print("Maximum Silhouette Score:", max(silhouette_scores))


# In[89]:


"""
Q2: Part 7: 

The nice part of using a GMM is that now, once you believe you selected the right number of 
components in your model, you can generate points from a GMM. The reason is that it is 
relatively simple to generate points from a Gaussian distribution (see the command numpy.random.normal) 
and therefore from a GMM. Generate 500 points from your GMM, does it seems to you they are 
reproducing the two moons?
"""


# Generate 500 points from the trained GMM
generated_points = gmm.sample(n_samples=500)
# Plot the original data points
plt.scatter(data_points[:, 0], data_points[:, 1], c='blue', label='Original Data', alpha=0.5)

# Plot the generated points
plt.scatter(generated_points[0][:, 0], generated_points[0][:, 1], c='red', label='Generated Data', alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Original Data and Generated Data')
plt.legend()
plt.grid(True)
plt.show()


# In[90]:


"""
Q3: GMM on the MNIST dataset as a generative model. Do for the MNIST data set what you did for the
two-moon example. In this data set each sample point (representing a hand written digit, represents a figure)
is defined in a high dimensional space (whose dimension is given by the number of pixels used to represent the
figure). It is therefore advisable to reduce the dimensionality of the space using PCA. I believe that the number
of components should be anything between 30 and 100, you should justify the number you pick by experiments
or by defining (and justfying) a metric that you believe is relevant for the problem.
The overall goal is to build a GMM that can generate new handwritten digits:

Q3 Part 1: Perform PCA on the original dataset. Justify the number of components you are choosing.

Answer: See word doc

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # This is for Principal Component Analysis
from sklearn.datasets import load_digits  # Importing load_digits
import time # for mesuaring execution time

# Load digits dataset
digits = load_digits()  # Using load_digits to load the digits dataset
X = digits.data / 16  # normalize pixel values
y = digits.target
# The reason why we normalize the pixel values here is because then we can ensure that 
# all the pixel values are within a consistent scale. Also, working with pixel values that 
# are in the range of 0 to 1 helps prevent numerical instabilities that can occur 
# when dealing with large values. 

# Measure execution time for PCA and GMM fitting
start_time = time.time()

# Perform PCA
n_components = 64 
# we're choosing 64 components bc dataset consists of 8x8 images. 
# Each pixel in the image corresponds to a feature, so the total 
# number of features is 64 (8 pixels wide by 8 pixels tall), hence 
# the choice of 64 components for PCA. 


pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
# This part is done to reduce the dimensionality of the dataset to 64 components.
# pca.fit_transform(X) helps fit the PCA model to the data and transform it.


# Calculate and print the cumulative explained variance ratio
explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
print(f"Explained variance with {n_components} components: {explained_variance_ratio * 100:.2f}%")
# The explained variance ratio tells us how much of the total variance is captured by the selected components.
# In this case, using 64 components should capture 100% of the variance because we have 64 features.
# Capturing 100% of the variance means that we are retaining all the info from the original dataset. 
# Thus no information is lost. While it's true that capturing 100% of the variance can include 
# capturing noise and other irrelevant information, no such issue has been noticed here.

# Fit GMM
n_components_gmm = 600  # Number of components for GMM
gmm = GaussianMixture(n_components=n_components_gmm)
gmm.fit(X_pca)
# The choice of 600 components for the GMM was made after experimentation 
# and consideration of multiple factors. Increasing the number of components 
# allows the model to capture a wider range of patterns and variations present 
# in the data, leading to clearer and more detailed images. While a higher number 
# of components may increase computational complexity and training time, it was 
# found that the trade-off was acceptable, resulting in both improved image 
# quality and manageable runtime.

end_time = time.time()
execution_time = end_time - start_time
print("Execution time for PCA and GMM fitting: {:.2f} seconds".format(execution_time))


# Generate new digits
n_samples = 10  # Number of samples to generate
generated_samples_pca = gmm.sample(n_samples)[0]
generated_samples = pca.inverse_transform(generated_samples_pca)

# Plot generated digits
plt.figure(figsize=(10, 5))
for i in range(n_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_samples[i].reshape(8, 8), cmap='gray')  # Adjusted to 8x8 pixels for load_digits
    plt.title('Generated Digit')
    plt.axis('off')
plt.show()


# In[91]:


"""
Q3 Part 2: 
Build a GMM as you did in part one of this final. Train first a GMM without looking at the labels
(considering it as a density estimator) and then using the labels. Choose what you believe is the best
in between the two models you obtain.

Answer: See word doc
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture
import time

# Load digits dataset
start_time = time.time()
digits = load_digits()
X = digits.data / 16.0  # normalize pixel values
y = digits.target

# Perform PCA
n_components = 64
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Train an unsupervised GMM
gmm_unsupervised = GaussianMixture(n_components=500, random_state=0)
gmm_unsupervised.fit(X_pca)

# Train a supervised GMM
pca_transformed_data_with_labels = np.hstack((X_pca, y.reshape(-1, 1)))
gmm_supervised = GaussianMixture(n_components=500, random_state=0)
gmm_supervised.fit(pca_transformed_data_with_labels)

# Calculate BIC scores
bic_unsupervised = gmm_unsupervised.bic(X_pca)
bic_supervised = gmm_supervised.bic(pca_transformed_data_with_labels)

print("Unsupervised GMM BIC:", bic_unsupervised)
print("Supervised GMM BIC:", bic_supervised)

# Choose the best model based on BIC
if bic_unsupervised < bic_supervised:
    best_model = gmm_unsupervised
    print("Best model: Unsupervised GMM")
else:
    best_model = gmm_supervised
    print("Best model: Supervised GMM")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))


# In[92]:


"""
Q3 Part 3: 
For each class of the dataset (that is characterized by 10 classes each one representing a digit), find the
GMM components belonging to that class and plot their means. If you did things correctly, each mean
should be identifiable as a digit.

Answer: See word doc

"""

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Step 2: Dimensionality Reduction with PCA
n_components = 64  
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Step 3: Train GMM
gmm = GaussianMixture(n_components=10, random_state=0)  # 10 components, one for each digit class
gmm.fit(X_pca)

# Step 4: Assign labels based on maximum probability
y_pred = gmm.predict(X_pca)

# Step 5: Visualize Means
digit_means = []
for i in range(10):
    indices = np.where(y == i)[0]  # Get indices of original digit labels
    mean_digit = np.mean(X_pca[indices], axis=0)
    digit_means.append(mean_digit)

# Step 6: Plot Means
digit_means_original_space = pca.inverse_transform(digit_means)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digit_means_original_space[i].reshape(8, 8), cmap='gray')
    ax.set_title(str(i))
    ax.axis('off')

plt.show()

import matplotlib.pyplot as plt


# In[93]:


"""
Q3 Part 4: 
Sample the GMM and plot some of these samples, do they look like hand written digits?
Answer: See word doc

"""
n_samples = 16
samples, _ = gmm.sample(n_samples)

# Transform the samples back to the original data space
samples_original_space = pca.inverse_transform(samples)

# Plot the samples
fig, axes = plt.subplots(1, n_samples, figsize=(10, 1))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples_original_space[i].reshape(8, 8), cmap='gray')
    ax.axis('off')

plt.show()


# In[94]:


"""
Q3 Part 5: 
Another way to check whether your GMM is performing well as a classifier is to compare its performance
by using a different classifier, like, for instance, a boosted gradient tree. If you generate points from the
GMM and assign them the label based on which component the point is coming from (remember that
each component is assigned to a given class), does your tree classify that point correctly?
Answer: See word doc

"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Step 1: Split original data into training and testing sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 2: Train Gradient Boosting Classifier on original data
gb_classifier_orig = GradientBoostingClassifier(random_state=0)
gb_classifier_orig.fit(X_train_orig, y_train_orig)

# Step 3: Evaluate Performance on original testing data
predicted_labels_orig = gb_classifier_orig.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test_orig, predicted_labels_orig)
print("Accuracy of Gradient Boosting Classifier on original testing data:", accuracy_orig)

# Step 4: Generate synthetic data from GMM
def generate_data_from_gmm(gmm, num_samples):
    X, _ = gmm.sample(num_samples)
    return X

# Step 5: Assign labels based on GMM components
def assign_labels_based_on_component(gmm, data):
    labels = gmm.predict(data)
    return labels

# Generate synthetic data from GMM
gmm = GaussianMixture(n_components=10, random_state=0)
gmm.fit(X)  # Assuming you have some data X
num_samples = 1000
generated_data = generate_data_from_gmm(gmm, num_samples)
true_labels = assign_labels_based_on_component(gmm, generated_data)

# Step 6: Split generated data into training and testing sets
X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(generated_data, true_labels, test_size=0.2, random_state=0)

# Step 7: Train Gradient Boosting Classifier on generated data
gb_classifier_gen = GradientBoostingClassifier(random_state=0)
gb_classifier_gen.fit(X_train_gen, y_train_gen)

# Step 8: Evaluate Performance on generated testing data
predicted_labels_gen = gb_classifier_gen.predict(X_test_gen)
accuracy_gen = accuracy_score(y_test_gen, predicted_labels_gen)
print("Accuracy of Gradient Boosting Classifier on generated testing data:", accuracy_gen)

