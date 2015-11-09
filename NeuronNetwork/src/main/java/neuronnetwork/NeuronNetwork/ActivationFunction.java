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
	 * Method returning function value in the point
	 * @param u Value of an inner function
	 * @return
	 */
	public double get(double u);
	
	/**
	 * Method returning derivative of the function
	 * @param u Value of an inner function
	 * @return
	 */
	public double getDerivative(double u); 
}
