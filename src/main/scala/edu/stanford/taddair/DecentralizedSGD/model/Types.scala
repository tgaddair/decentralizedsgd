package edu.stanford.taddair.DecentralizedSGD.model

import breeze.linalg.{DenseMatrix, DenseVector}


object Types {

  type Activation = DenseVector[Double]
  type ActivationFunction = DenseVector[Double] => DenseVector[Double]
  type LayerWeight = DenseMatrix[Double]
  type SparseLayerWeight = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Double]]
  type Delta = DenseVector[Double]

}
