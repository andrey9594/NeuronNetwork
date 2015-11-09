package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * ActivationFunction for M-P perceptron
 */
public class NeuronKernelFunctionSum implements NeuronKernelFunction {

	public Double get(double x[], double C, double w[]) {
		double sum = 0.;
		for (int i = 0; i < x.length; i++)
			sum += x[i] * w[i];
		return sum;
	}

}
