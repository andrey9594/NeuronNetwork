package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Sigmoid hyperbolic tan function
 */
public class FunctionSigmoidTan implements ActivationFunction {

	public double get(double u) {
		return Math.tanh(u);
	}

	public double getDerivative(double u) {
		return u;
	}
}
