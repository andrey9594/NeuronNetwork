package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Sigmoid logistic function	
 */
public class FunctionSigmoidLogistic implements ActivationFunction {

	public double get(double u) {
		return 1. / (1 + Math.pow(Math.E, -u));
	}
	
	public double getDerivative(double u) {
		return u;
	}

}
