package edu.stanford.taddair.DecentralizedSGD.actors.decentralized

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DCrossValidator.StartTraining
import edu.stanford.taddair.DecentralizedSGD.actors.decentralized.DMaster._
import edu.stanford.taddair.DecentralizedSGD.model.Types.ActivationFunction
import edu.stanford.taddair.DecentralizedSGD.model.Example

object DMaster {

  case class ValidationDone(modelId: Int, meanSquaredError: Double)

  case object Start

  case object JobDone

}

/**
  * The master actor of the DistBelief implementation.
  * @param dataSet The data set to be used for training.
  * @param dataPerReplica The number of data points to be used in a data shard.
  * @param layerDimensions The number of neural units in each layer of the neural network model.
  * @param activation The activation function.
  * @param activationDerivative The derivative of the activation function.
  * @param learningRates The learning rate hyperparameters for parameter updates.
  */
class DMaster(dataSet: Seq[Example],
              dataPerReplica: Int,
              layerDimensions: Seq[Int],
              activation: ActivationFunction,
              activationDerivative: ActivationFunction,
              tau: Double,
              learningRates: Seq[Double]) extends Actor with ActorLogging {

  val numLayers = layerDimensions.size

  // Cross validator for every set of possible hyperparameters
  val crossValidatorActors: Array[ActorRef] = new Array[ActorRef](learningRates.size)
  for ((lr, i) <- learningRates.zipWithIndex) {
    crossValidatorActors(i) = context.actorOf(Props(new DCrossValidator(
      modelId = i,
      dataSet = dataSet,
      dataPerReplica = dataPerReplica,
      tau = tau,
      learningRate = lr,
      numLayers = numLayers,
      layerDimensions = layerDimensions,
      activation = activation,
      activationDerivative = activationDerivative
    )))
  }

  var numModelsFinished = 0
  var bestModelId = 0
  var bestError = Double.PositiveInfinity
  def receive = {
    case Start => crossValidatorActors.foreach(_ ! StartTraining)

    case ValidationDone(id, meanSquaredError) => {
      numModelsFinished += 1
      if (meanSquaredError < bestError) {
        bestModelId = id
        bestError = meanSquaredError
      }

      log.info(s" model ${id} completed with mean squared error ${meanSquaredError}")
      log.info(s" ${numModelsFinished} models finished of ${crossValidatorActors.size}")

      if (numModelsFinished == crossValidatorActors.size) {
        log.info(s"Best Model: ${id}, learning rate = ${learningRates(id)}, mean squared error = ${bestError}")
        context.parent ! JobDone
        context.stop(self)
      }
    }
  }
}
