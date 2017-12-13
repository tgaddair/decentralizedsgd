package edu.stanford.taddair.DecentralizedSGD.actors.centralized

import akka.actor.{Actor, ActorRef, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.CrossValidator.FinishedTraining
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.DataShard.{FetchParameters, ReadyToProcess}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Layer.{DoneFetchingParameters, ForwardPass, MyChild}
import edu.stanford.taddair.DecentralizedSGD.model.Example
import edu.stanford.taddair.DecentralizedSGD.model.Types.ActivationFunction


object DataShard {

  case object ReadyToProcess

  case object FetchParameters

}

/**
 * The data shard actor for the DistBelief implementation.
 * @param shardId Unique Id for this data shard.
 * @param trainingData The training data for this shard.
 * @param activation The activation function.
 * @param activationDerivative The derivative of the activation function.
 * @param parameterShards actorRefs for the parameter shards that hold the model parameters.
 */
class DataShard(shardId: Int,
                trainingData: Seq[Example],
                activation: ActivationFunction,
                activationDerivative: ActivationFunction,
                parameterShards: Seq[ActorRef],
                outputActor: ActorRef) extends Actor {

  val numLayers = parameterShards.size

  //parameter shard corresponding to each layer
  val trainingDataIterator = trainingData.toIterator

  //create layer actors for this shard's model replica
  val layers: Array[ActorRef] = new Array[ActorRef](numLayers)

  for (l <- 0 until numLayers) {

    layers(l) = context.actorOf(Props(new Layer(
      replicaId = shardId,
      layerId = l,
      activationFunction = activation,
      activationFunctionDerivative = activationDerivative,
      parentLayer = if (l > 0) Some(layers(l - 1)) else None, //parent layer actor
      parameterShardId = parameterShards(l),
      outputAct = Some(outputActor))) //layer needs access to its parameter shard to read from and update
    )

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
    case ReadyToProcess => {

      layers.foreach(_ ! FetchParameters)
      context.become(waitForAllLayerUpdates)
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
