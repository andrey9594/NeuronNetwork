package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Type of neuron activation function
 * Sigmoidal for M-P Perceptron
 * and for RFB: e^((x-c)^2 / 2*sigma^2)
 */
public interface ActivationFunction {
	/**
	 * 
	 * @param u Value of an inner function
	 * @return
	 */
	public double get(double u);
	
	/**
	 * 
	 * @param u Value of an inner function
	 * @return
	 */
	public double getDerivative(double u); 
}
