
In this repo we explore many methods for encoding timeseries (and some tabular) data, for prediction in regression. Considered methods include components:
- Basic architectures: multi-layer perceptron (MLP), CNN, Barlow twins, Long Short Term Memory (LSTM)
- Autoencoders (AE): [variational] AE (VAE), Conditional VAE (C-VAE), β-VAE
- Timeseries encoders: TS2Vec, TimeVAE, MOMENT

Plus hyperparam search, benchmarking, evaluation metrics

More recently, I pivoted to a topological-geometric approach (see `geo` folder), hoping to gain novel insights and inductive biases. I used a combination of the above components, plus the following (some are borrowed from others, some are homemade): spherical MLP/LSTM encoders, toroidal MLP/LSTM encoder, von Mises-Fisher sampling, manifolds ...

![Project idea](images/project_idea.png)
<!-- ![Topo Diagram](images/topo_diagram.png) -->
![Latent Progression](images/z_progression.png)
![Point cloud](images/point_cloud.png)
![Persistence diagrams](images/persistence_diagrams.png)
![Persistence images](images/persistence_images.png)
![Berri curves](images/betti_curves.png)

<!-- ![Torus](images/torus.png)
![Torus Persistence](images/torus_persistence.png)
![Torus Betti Curves](images/torus_betti.png) -->

The topmost image shows the architecture of the current idea.
The bottom images show a noisy torus structure, along with its persistence diagram, and Betti curves.
