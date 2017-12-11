Decentralized SGD
===============

This is a framework for performing distributed training of machine learning models over a cluster of
machines in parallel.  It implements two methods for performing this training:

1. Centralized training based on the DistBelief framework
2. Decentralized training based on the work by Strom

The Akka framework for using an actor-based model is used to implement these methods, and is based on the work
done by Alex Minnaar (https://github.com/alexminnaar/AkkaDistBelief).  We extend his centralized implementation
by adding cross validation for model selection, and implement the decentralized implementation alongside it.
