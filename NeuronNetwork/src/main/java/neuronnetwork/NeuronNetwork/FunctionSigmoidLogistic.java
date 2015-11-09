package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Sigmoid logistic function	
 */
public class FunctionSigmoidLogistic implements ActivationFunction {
	/**
	 * alpha paremetr in function 1 / (1 + e ^ {-alpha * x})
	 */
	private double alpha = 1.;
	
	/**
	 * The empty constructor: alpha is remaining by default
	 */
	public FunctionSigmoidLogistic() { }
	
	/**
	 * The constructor with alpha parameter
	 * @param alpha Alpha parameter in the logistic function
	 */
	public FunctionSigmoidLogistic(double alpha) {
		this.alpha = alpha;
	}

	public double get(double u) {
		return 1. / (1 + Math.pow(Math.E, -u * alpha));
	}
	
	public double getDerivative(double u) {
		double f = this.get(u);
		return alpha * f * (1 - f);
	}

}
