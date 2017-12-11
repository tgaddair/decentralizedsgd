package edu.stanford.taddair.DecentralizedSGD.actors.centralized

import akka.actor.{Actor, ActorLogging}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Layer.Gradient
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.ParameterShard.{LatestParameters, ParameterRequest}
import edu.stanford.taddair.DecentralizedSGD.model.Types.LayerWeight


object ParameterShard {

  case class ParameterRequest(dataShardId: Int, layerId: Int)

  case class LatestParameters(weights: LayerWeight)

}

/**
 * The parameter shard actor for the DistBelief implementation.
 * @param shardId Unique Id for this parameter shard.
 * @param learningRate Learning rate for parameter updates.
 * @param initialWeight Initial random weight matrix.
 */
class ParameterShard(shardId: Int,
                     learningRate: Double,
                     initialWeight: LayerWeight) extends Actor with ActorLogging {

  //weights initialize randomly
  var latestParameter: LayerWeight = initialWeight

  def receive = {

    //A layer corresponding to this shardId in some model replica has requested the latest version of the parameters.
    case ParameterRequest(shardId, layerId) => {
      log.info(s"layer ${layerId} weights read by model replica ${shardId}")
      context.sender() ! LatestParameters(latestParameter)
    }

    /*
    A layer corresponding to this shardId in some model replica has computed a gradient, so we must update our
    parameters according to this gradient.
    */
    case Gradient(g, replicaId, layerId) => {
      log.info(s"layer ${layerId} weights updated by model replica ${replicaId}")
      latestParameter = latestParameter + g.t * learningRate
    }

  }

}
