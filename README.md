# gmm-clustering-classification
Exploring Gaussian Mixture Models for clustering, classification, and digit generation using Python and scikit-learn.

🧠 Gaussian Mixture Models: Clustering, Classification & Generation
This project explores Gaussian Mixture Models (GMMs) through both synthetic and real-world datasets — starting with the classic two moons and scaling up to the MNIST handwritten digits dataset.

Along the way, I applied concepts like dimensionality reduction, unsupervised learning, generative modeling, and model selection techniques. This wasn't just about getting code to run — it was about understanding why and how these models work in different settings.

🔍 What this project covers
1. Clustering with GMM on Two Moons
Used make_moons() to simulate non-linear cluster data

Trained GMM with 2 components to fit the data

Visualized results using decision boundaries and level sets

Compared clustering performance with different covariance types and component counts

2. Model Selection & Tuning
Applied GridSearchCV to find the optimal number of components and covariance type using BIC

Used Silhouette Score to evaluate cluster compactness and separation

Prevented overfitting through cross-validation and test-set validation

3. Posterior-Based Classification
Used GMM to assign class probabilities (P(Y|X=x))

Implemented a custom strategy to map GMM components to class labels

Classified new points by evaluating component probabilities

4. GMM as a Generative Model
Sampled synthetic points from trained GMMs

Compared generated vs. real distributions (for both moons and digits)

5. GMM on MNIST
Applied PCA to reduce dimensionality of 64-pixel digit data

Trained GMM with up to 600 components to model digit variation

Generated new digit images from the model

Compared supervised vs. unsupervised GMMs using BIC

Evaluated GMM-generated labels using a Gradient Boosting Classifier

💻 Technologies & Libraries
Python

scikit-learn (GMM, PCA, GridSearchCV, silhouette score, classification models)

matplotlib and seaborn for visualization

numpy for matrix operations and probability calculations

📈 What I practiced & learned
How to tune unsupervised models using real validation metrics

Interpreting GMM results both visually and numerically

Using PCA to reduce dimensionality without losing essential patterns

When to use unsupervised learning vs supervised methods — and how to compare them

How GMMs can be applied beyond clustering — for classification and generation
