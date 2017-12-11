package edu.stanford.taddair.DecentralizedSGD.actors.decentralized

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DCrossValidator.{FinishedTraining, StartTraining}
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DDataShard.ReadyToProcess
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DMaster.ValidationDone
import edu.stanford.taddair.DecentralizedSGD.model.Types._
import edu.stanford.taddair.DecentralizedSGD.model.{Example, NeuralNetworkOps}


object DCrossValidator {

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
class DCrossValidator(modelId: Int,
                      dataSet: Seq[Example],
                      dataPerReplica: Int,
                      tau: Double,
                      learningRate: Double,
                      numLayers: Int,
                      layerDimensions: Seq[Int],
                      activation: ActivationFunction,
                      activationDerivative: ActivationFunction) extends Actor with ActorLogging {

  // Create actors for each data shard (replica).
  // Each replica needs to know about all parameter shards in the centralized version
  // because they will be reading from them and updating them.
  val dataShards = dataSet.grouped(dataPerReplica).toSeq
  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DDataShard(
      shardId = dataShard._2,
      trainingData = dataShard._1,
      tau = tau,
      learningRate = learningRate,
      activation = activation,
      activationDerivative = activationDerivative,
      numLayers = numLayers - 1,
      layerDimensions = layerDimensions: Seq[Int],
      seed = scala.util.Random.nextInt())))
  }
  log.info(s"model ${modelId}: ${dataShards.size} data shards initiated!")

  var numShardsFinished = 0
  var squaredErrorSum = 0.0
  var nSum = 0.0
  def receive = {
    case StartTraining => dataShardActors.foreach(_ ! ReadyToProcess(dataShardActors))

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
