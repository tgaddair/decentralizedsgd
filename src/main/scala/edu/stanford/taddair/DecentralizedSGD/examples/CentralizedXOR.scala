package edu.stanford.taddair.DecentralizedSGD.examples

import akka.actor.{Actor, ActorLogging, Props}
import breeze.linalg.DenseVector
import breeze.numerics._
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Master
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Master.{JobDone, Start}
import edu.stanford.taddair.DecentralizedSGD.model.Example

import scala.util.Random

/*
An example using DistBelief to learn the non-linear XOR function

A | B | Output
---------------
0 | 0 | 1
0 | 1 | 0
1 | 0 | 0
1 | 1 | 1

 */

class CentralizedXOR extends Actor with ActorLogging {

  val random = new Random

  val possibleExamples = Seq(
    Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
    , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0))
  )

  //generate 50000 training examples
  val trainingSet = (1 to 50000).foldLeft(Seq[Example]()) { (a, c) =>
    a :+ possibleExamples(random.nextInt(possibleExamples.size))
  }

  //create 25 model replicas each training 2000 data points in parallel
  val DistBeliefMaster = context.actorOf(Props(new Master(
    dataSet = trainingSet,
    dataPerReplica = 2000,
    layerDimensions = Seq(2, 2, 1),
    activation = (x: DenseVector[Double]) => x.map(el => sigmoid(el)),
    activationDerivative = (x: DenseVector[Double]) => x.map(el => sigmoid(el) * (1 - sigmoid(el))),
    learningRates = Seq(0.3, 0.5, 0.7))
  )
  )

  DistBeliefMaster ! Start

  def receive = {
    case JobDone => log.info("Finished Computing XOR Example!!!!!")
  }


}
