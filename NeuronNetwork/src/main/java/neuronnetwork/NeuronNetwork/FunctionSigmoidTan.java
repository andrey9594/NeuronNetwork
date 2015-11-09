package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Sigmoid hyperbolic tan function
 */
public class FunctionSigmoidTan implements ActivationFunction {
	/**
	 * alpha paremetr in function th{alpha * x}
	 */
	private double alpha = 1.;
	
	/**
	 * The empty constructor: alpha is remaining by default
	 */
	public FunctionSigmoidTan() { }
	
	/**
	 * The constructor with alpha parameter
	 * @param alpha Alpha parameter in the tanh function
	 */
	public FunctionSigmoidTan(double alpha) {
		this.alpha = alpha;
	}
	
	public double get(double u) {
		return Math.tanh(alpha * u);
	}

	public double getDerivative(double u) {
		double f = this.get(u);
		return alpha * (1 - f * f);
	}
}
