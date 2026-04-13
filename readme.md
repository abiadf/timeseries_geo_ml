
In this repo we explore many methods for encoding timeseries (and some tabular) data, for prediction in regression. Considered methods include components:
- Basic architectures: multi-layer perceptron (MLP), CNN, Barlow twins, Long Short Term Memory (LSTM)
- Autoencoders (AE): [variational] AE (VAE), Conditional VAE (C-VAE), β-VAE
- Timeseries encoders: TS2Vec, TimeVAE, MOMENT

Plus hyperparam search, benchmarking, evaluation metrics

More recently, I pivoted to a topological-geometric approach (see `geo` folder), hoping to gain novel insights and inductive biases. I used a combination of the above components, plus the following (some are borrowed from others, some are homemade): spherical MLP/LSTM encoders, toroidal MLP/LSTM encoder, von Mises-Fisher sampling, manifolds ...

![Topo Diagram](images/topo_diagram.png)
![Torus](images/torus.png)
![Torus Persistence](images/torus_persistence.png)
![Torus Betti Curves](images/torus_betti.png)

Images showing a noisy torus structure, along with its persistence diagram, and Betti curves.
