package edu.stanford.taddair.DecentralizedSGD.actors.decentralized

import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.{DenseMatrix, DenseVector}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.OutputActor.Output
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DDataShard.{FetchParameters, LayerParameterUpdate, ReadyToProcess}
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DLayer._
import edu.stanford.taddair.DecentralizedSGD.model.NeuralNetworkOps._
import edu.stanford.taddair.DecentralizedSGD.model.Types._

object DLayer {

  case class DoneFetchingParameters(layerId: Int, squaredError: Double, n: Int)

  case class Gradient(g: DenseMatrix[Double], replicaId: Int, layerId: Int)

  case class ForwardPass(inputs: DenseVector[Double], target: DenseVector[Double])

  case class BackwardPass(deltas: Delta)

  case class MyChild(ar: ActorRef)

  case class ParameterUpdate(weights: SparseLayerWeight)

}

/**
  * An Akka actor representing a layer in a DistBelief neural network replica.
  * @param replicaId A unique model replica identifier.
  * @param layerId The layer of the replica that this actor represents
  * @param activationFunction Neural network activation function.
  * @param activationFunctionDerivative Derivative of activation function.
  * @param parentLayer actorRef of this actor's parent layer (may not exist if this is an input layer).
  * @param outputAct actorRef of output actor.
  */
class DLayer(replicaId: Int,
             layerId: Int,
             tau: Double,
             learningRate: Double,
             activationFunction: ActivationFunction,
             activationFunctionDerivative: ActivationFunction,
             parentLayer: Option[ActorRef],
             outputAct: Option[ActorRef],
             initialWeight: LayerWeight) extends Actor with ActorLogging {


  var latestWeights: LayerWeight = initialWeight
  var gradientResidual: DenseMatrix[Double] = new DenseMatrix[Double](latestWeights.rows, latestWeights.cols)
  var activations: Activation = _
  var activatedInput: Activation = _
  var childLayer: Option[ActorRef] = None

  var squaredError = 0.0
  var n = 0
  var dataShards: Seq[ActorRef] = Nil

  def receive = {

    //Before we process a data point, we must update the parameter weights for this layer
    case FetchParameters(shards) => {
      dataShards = shards
      context.parent ! DoneFetchingParameters(layerId, squaredError, n)
//      parameterShardId ! ParameterRequest(replicaId, layerId)
//      context.become(waitForParameters)
    }

    //If this layer has a child, its identity will be sent.
    case MyChild(ar) => childLayer = Some(ar)

    case ForwardPass(inputs, target) => {
      activatedInput = parentLayer match {
        case Some(p) => DenseVector.vertcat(DenseVector(1.0), activationFunction(inputs))
        case _ => inputs
      }

      //compute outputs given current weights and received inputs, then pass them through activation function
      val outputs = computeLayerOutputs(activatedInput, latestWeights)
      val activatedOutputs = activationFunction(outputs)

      activations = parentLayer match {
        case Some(p) => DenseVector.vertcat(DenseVector(1.0), inputs)
        case _ => inputs
      }

      childLayer match {

        //If there is a child layer, send it the outputs with an added bias
        case Some(nextLayer) => {
          nextLayer ! ForwardPass(outputs, target)
        }

        //if this is the final layer of the neural network, compute prediction error and send the result backwards.
        case _ => {

          //compute deltas which we can use to compute the gradients for this layer's weights.
          val deltas = computePredictionError(activatedOutputs, target)
          val gradient = computeGradient(deltas, activatedInput)

          squaredError = computeSumSquaredError(deltas)
          n = deltas.length

          //send gradients for updating in the parameter shard actor
          gradientResidual = gradientResidual + gradient.t

          val update = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Double]]()
          for (r <- 0 until gradientResidual.rows) {
            for (c <- 0 until gradientResidual.cols) {
              val v = gradientResidual(r, c)
              if (v > tau) {
                if (!update.contains(r)) {
                  update.put(r, scala.collection.mutable.Map[Int, Double]())
                }
                update(r).put(c, v)
                gradientResidual.update(r, c, 0)
              } else if (v < -tau) {
                if (!update.contains(r)) {
                  update.put(r, scala.collection.mutable.Map[Int, Double]())
                }
                update(r).put(c, v)
                gradientResidual.update(r, c, 0)
              }
            }
          }

          latestWeights = latestWeights + gradient.t * learningRate

          if (update.nonEmpty) {
            for (shard <- dataShards) {
              shard ! LayerParameterUpdate(layerId, update)
            }
          }
//          parameterShardId ! Gradient(gradient, replicaId, layerId)

          //compute the deltas for this parent layer (there must be one if this is the output layer)
          val parentDeltas = computeDeltas(deltas, activations, latestWeights, activationFunctionDerivative)
          context.sender() ! BackwardPass(parentDeltas)

          //If this is the last layer then send the predictions to the output actor
          outputAct.get ! Output(replicaId, target, activatedOutputs)
        }
      }

    }

    case BackwardPass(childDeltas) => {
      //compute gradient of layer weights given deltas from child layer and activations from forward pass and
      //send the resulting gradient to the parameter shard for updating.
      val gradient = computeGradient(childDeltas, activatedInput)

      val update = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Double]]()
      for (r <- 0 until gradientResidual.rows) {
        for (c <- 0 until gradientResidual.cols) {
          val v = gradientResidual(r, c)
          if (v > tau) {
            if (!update.contains(r)) {
              update.put(r, scala.collection.mutable.Map[Int, Double]())
            }
            update(r).put(c, v)
            gradientResidual.update(r, c, 0)
          } else if (v < -tau) {
            if (!update.contains(r)) {
              update.put(r, scala.collection.mutable.Map[Int, Double]())
            }
            update(r).put(c, v)
            gradientResidual.update(r, c, 0)
          }
        }
      }

      latestWeights = latestWeights + gradient.t * learningRate

      if (update.nonEmpty) {
        for (shard <- dataShards) {
          shard ! LayerParameterUpdate(layerId, update)
        }
      }
//      parameterShardId ! Gradient(gradient, replicaId, layerId)

      parentLayer match {

        //If there is a parent layer, compute deltas for this layer and send them backwards.  We remove the delta
        //corresponding to the bias unit because it is not connected to anything in the parent layer thus it should
        //not affect its gradient.
        case Some(previousLayer) => {
          val parentDeltas = computeDeltas(childDeltas, activations, latestWeights, activationFunctionDerivative)
          previousLayer ! BackwardPass(parentDeltas(1 to -1))
        }

        //If this is the first layer, let data shard know we are ready to update weights and process another data point.
        case _ => {
          context.parent ! ReadyToProcess(dataShards)
        }
      }
    }

    case ParameterUpdate(weights) => {
      log.info(s"${replicaId} ${layerId}: parameter update")

      val newWeights = latestWeights.copy
      for ((row, colWeight) <- weights) {
        for ((col, g) <- colWeight) {
          val v = latestWeights(row, col) + g * learningRate
          newWeights.update(row, col, v)
//          log.info(s"${replicaId} ${layerId}: updated ${g} -> ${v}")
        }
      }
      latestWeights = newWeights

//      context.parent ! DoneFetchingParameters(layerId, squaredError, n)
//      context.unbecome()
    }
  }
}
