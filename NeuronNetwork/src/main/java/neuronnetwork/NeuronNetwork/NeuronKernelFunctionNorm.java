package neuronnetwork.NeuronNetwork;

/**
 * 
 * @author andrey
 * ActivationFunction for RBF
 */
public class NeuronKernelFunctionNorm implements NeuronKernelFunction {
	public Double get(double[] x, double C, double[] w) {
		double norm = 0.;
		for (int i = 0; i < x.length; i++) {
			norm += (x[i] - C) * (x[i] - C);
		}
		norm = Math.sqrt(norm);
		return norm;
	}

}	
