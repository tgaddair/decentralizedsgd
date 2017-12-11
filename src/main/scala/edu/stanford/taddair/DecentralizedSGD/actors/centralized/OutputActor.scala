package edu.stanford.taddair.DecentralizedSGD.actors.centralized

import akka.actor.{Actor, ActorLogging}
import breeze.linalg.DenseVector
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.OutputActor.Output

object OutputActor {

  case class Output(replicaId: Int, target: DenseVector[Double], output: DenseVector[Double])

}

/**
 * Actor that logs outputs and keeps track of the last predictions of each model replica
 */
class OutputActor extends Actor with ActorLogging {

  var latestOutputs: Map[Int, DenseVector[Double]] = Map.empty

  def receive = {

    case Output(replica, target, output) => {

      latestOutputs += (replica -> output)

      log.info(s"replica id ${replica}, output: ${output}, target ${target}")
    }


  }


}
