package neuronnetwork.NeuronNetwork;

/**
 * Identity function
 * For instance for MLP's input layer	
 */
public class FunctionIdentity implements ActivationFunction {
	
	@Override
	public double get(double u) {
		return u;
	}

	@Override
	public double getDerivative(double u) {
		return 0;
	}

}
