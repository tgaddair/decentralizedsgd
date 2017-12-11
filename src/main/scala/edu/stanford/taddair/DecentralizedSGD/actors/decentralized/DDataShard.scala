package edu.stanford.taddair.DecentralizedSGD.actors.decentralized

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.OutputActor
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DCrossValidator.FinishedTraining
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DDataShard.{FetchParameters, LayerParameterUpdate, ReadyToProcess}
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DLayer.{DoneFetchingParameters, ForwardPass, MyChild, ParameterUpdate}
import edu.stanford.taddair.DecentralizedSGD.model.{Example, NeuralNetworkOps}
import edu.stanford.taddair.DecentralizedSGD.model.Types._

import scala.collection.mutable.ListBuffer


object DDataShard {

  case class ReadyToProcess(dataShards: Seq[ActorRef])

  case class FetchParameters(dataShards: Seq[ActorRef])

  case class LayerParameterUpdate(layerId: Int, weights: SparseLayerWeight)

}

/**
  * The data shard actor for the DistBelief implementation.
  * @param shardId Unique Id for this data shard.
  * @param trainingData The training data for this shard.
  * @param activation The activation function.
  * @param activationDerivative The derivative of the activation function.
  */
class DDataShard(shardId: Int,
                 trainingData: Seq[Example],
                 tau: Double,
                 learningRate: Double,
                 activation: ActivationFunction,
                 activationDerivative: ActivationFunction,
                 numLayers: Int,
                 layerDimensions: Seq[Int],
                 seed: Int) extends Actor with ActorLogging {

  val outputActor = context.actorOf(Props(new OutputActor))

  //parameter shard corresponding to each layer
  val trainingDataIterator = trainingData.toIterator

  //create layer actors for this shard's model replica
  val layers: Array[ActorRef] = new Array[ActorRef](numLayers)

  for (l <- 0 until numLayers) {

    layers(l) = context.actorOf(Props(new DLayer(
      replicaId = shardId,
      layerId = l,
      tau = tau,
      learningRate = learningRate,
      activationFunction = activation,
      activationFunctionDerivative = activationDerivative,
      parentLayer = if (l > 0) Some(layers(l - 1)) else None, //parent layer actor
      outputAct = if (l == numLayers - 1) Some(outputActor) else None,
      initialWeight = NeuralNetworkOps.randomMatrixSeeded(layerDimensions(l + 1), layerDimensions(l) + 1, seed)
    )))

    //after each layer actor is created, let the previous layer know that its child is ready
    if (l > 0) layers(l - 1) ! MyChild(layers(l))
  }

  /*
  set to keep track of layers that have not yet been updated.
  Remove layerIds as the get updated with current versions of parameters.
  When set is empty, all layers are updated and we can process a new data point (also refill set at this point).
  */
  var layersNotUpdated = (0 until numLayers).toSet

  def receive = {
    /*
        if the minibatch has been successfully backpropagated, ask all model layers to update their parameters
        in order for the next data point to be processed.  Go into a waiting context until they have all been updated.
        */
    case ReadyToProcess(dataShards) => {
      val shardBuffer = new ListBuffer[ActorRef]
      for ((shard, i) <- dataShards.zipWithIndex) {
        if (shardId != i) {
          shardBuffer += shard
        }
      }
      val shards = shardBuffer.toList

      layers.foreach(_ ! FetchParameters(shards))
      context.become(waitForAllLayerUpdates)
    }

    case LayerParameterUpdate(layerId, weights) => {
      layers(layerId) ! ParameterUpdate(weights)
    }

  }

  var squaredErrorSum = 0.0
  var nSum = 0

  def waitForAllLayerUpdates: Receive = {
    case DoneFetchingParameters(layerId, squaredError, n) => {
      layersNotUpdated -= layerId
      squaredErrorSum += squaredError
      nSum += n

      //if all layers have updated to the latest parameters, can then process a new data point.
      if (layersNotUpdated.isEmpty) {
        if (trainingDataIterator.hasNext) {
          val dataPoint = trainingDataIterator.next()
          layers.head ! ForwardPass(dataPoint.x, dataPoint.y)
        }
        //If we have processed all of them then we are done.
        else {
          context.parent ! FinishedTraining(shardId, squaredErrorSum, nSum)
          context.stop(self)
        }

        layersNotUpdated = (0 until numLayers).toSet
        context.unbecome()
      }
    }
  }
}
