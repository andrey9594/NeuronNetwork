package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * Type of neuron inner function
 * for example: M-P perceptron: sum xi * wi
 * and for RBF: ||x - c||  	
 */
public interface NeuronKernelFunction {
	/**
	 * 
	 * @param x features
	 * @param C for RBF: the center (else unused)
	 * @param w weights or null if not need (in RBF for example)
	 * @return 
	 */
	public Double get(double x[], double C, double w[]); 
}
