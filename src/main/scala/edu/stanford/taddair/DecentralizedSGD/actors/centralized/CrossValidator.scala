package edu.stanford.taddair.DecentralizedSGD.actors.centralized

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.CrossValidator.{FinishedTraining, StartTraining}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.DataShard.ReadyToProcess
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Master.ValidationDone
import edu.stanford.taddair.DecentralizedSGD.model.Types._
import edu.stanford.taddair.DecentralizedSGD.model.{Example, NeuralNetworkOps}


object CrossValidator {

  case class FinishedTraining(dataShardId: Int, squaredError: Double, n: Int)

  case object StartTraining

}

/**
  * The cross validation actor used for evaluating a single configuration
  * of the model hyperparameters.
  *
  * @param modelId Unique Id for this model variant.
  * @param learningRate Learning rate for parameter updates.
  */
class CrossValidator(modelId: Int,
                     dataSet: Seq[Example],
                     dataPerReplica: Int,
                     learningRate: Double,
                     numLayers: Int,
                     layerDimensions: Seq[Int],
                     activation: ActivationFunction,
                     activationDerivative: ActivationFunction) extends Actor with ActorLogging {

  // Parameter shard actors for each network layer
  val parameterShardActors: Array[ActorRef] = new Array[ActorRef](numLayers - 1)
  for (i <- 0 to numLayers - 2) {
    parameterShardActors(i) = context.actorOf(Props(new ParameterShard(
      shardId = i,
      learningRate = learningRate,
      initialWeight = NeuralNetworkOps.randomMatrix(layerDimensions(i + 1), layerDimensions(i) + 1)
    )))
  }
  log.info(s"model ${modelId}: ${numLayers - 1} parameter shards initiated!")

  // Create actors for each data shard (replica).
  // Each replica needs to know about all parameter shards in the centralized version
  // because they will be reading from them and updating them.
  val dataShards = dataSet.grouped(dataPerReplica).toSeq
  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DataShard(
      shardId = dataShard._2,
      trainingData = dataShard._1,
      activation = activation,
      activationDerivative = activationDerivative,
      parameterShards = parameterShardActors)))
  }
  log.info(s"model ${modelId}: ${dataShards.size} data shards initiated!")

  var numShardsFinished = 0
  var squaredErrorSum = 0.0
  var nSum = 0.0
  def receive = {
    case StartTraining => dataShardActors.foreach(_ ! ReadyToProcess)

    case FinishedTraining(id, squaredError, n) => {
      numShardsFinished += 1
      squaredErrorSum += squaredError
      nSum += n

      log.info(s" model ${modelId}: shard completed with error ${squaredError}, ${n} samples")
      log.info(s" model ${modelId}: ${numShardsFinished} shards finished of ${dataShards.size}")

      if (numShardsFinished == dataShards.size) {
        val meanError = squaredErrorSum / n
        context.parent ! ValidationDone(modelId, meanError)
        context.stop(self)
      }
    }
  }
}
